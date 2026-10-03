use std::path::{Path, PathBuf};

use burn::{Dispatch, DispatchDevice};
use rverdict_core::{
    Answer, OrderedMap, Rendered, RenderedKind, Request, Response, Truncation, Usage, cache_root,
    render, state_text,
};
use rverdict_model::{
    DecisionModel, EncoderConfig, OptionAttention, PackedSequence, Precision, build_input,
    load_decision_pytorch, load_decision_safetensors, save_decision_safetensors,
};

use crate::EngineError;
use crate::checkpoint::{Checkpoint, Settings, Weights};
use crate::device::SelectedBackend;
use crate::pack::{Cut, Packer};

/// Padded tokens per forward pass, and padded attention-mask cells, so long
/// batches split instead of exhausting memory on small devices.
const BATCH_TOKENS: usize = 16_384;
const BATCH_MASK_CELLS: usize = 64 << 20;

/// A loaded checkpoint on a selected backend, ready to answer requests.
pub struct Engine {
    model: DecisionModel<Dispatch>,
    device: DispatchDevice,
    backend: String,
    config: EncoderConfig,
    packer: Packer,
    settings: Settings,
    name: String,
}

/// Where a converted copy of a PyTorch checkpoint lives. Only Hugging Face
/// cache blobs, which are named by their content hash, are converted, so a
/// cached copy can never belong to different weights.
fn converted(pytorch: &Path) -> Option<PathBuf> {
    let blob = std::fs::canonicalize(pytorch).ok()?;
    let hash = blob.file_name()?.to_str()?;
    (hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit())).then(|| {
        cache_root()
            .join("converted")
            .join(format!("{hash}.safetensors"))
    })
}

fn save_converted(model: &DecisionModel<Dispatch>, path: &Path) -> Result<(), EngineError> {
    let dir = path.parent().expect("converted paths have a parent");
    std::fs::create_dir_all(dir).map_err(|source| EngineError::Io {
        path: dir.to_owned(),
        source,
    })?;
    let partial = path.with_extension(format!("partial-{}", std::process::id()));
    save_decision_safetensors(model, &partial)?;
    std::fs::rename(&partial, path).map_err(|source| EngineError::Io {
        path: path.to_owned(),
        source,
    })
}

/// One question's packed rows: its own, and for an uncriteria'd `noul` the
/// same question asked of an empty state.
struct Planned {
    id: String,
    rendered: Rendered,
    row: usize,
    null_row: Option<usize>,
    state_tokens: usize,
    cut: Option<Cut>,
}

impl Engine {
    pub fn load(
        checkpoint: &Checkpoint,
        backend: SelectedBackend,
        precision: Precision,
    ) -> Result<Self, EngineError> {
        let device = backend.device;
        let mut model = checkpoint.config.init_decision_model::<Dispatch>(&device);
        match &checkpoint.weights {
            Weights::Safetensors(path) => {
                load_decision_safetensors(&mut model, path, precision)?;
            }
            // PyTorch pickles are slow to parse and load only as f32, so they
            // are converted to safetensors once and read from that copy.
            Weights::PyTorch(path) => match converted(path) {
                Some(cached) if cached.exists() => {
                    load_decision_safetensors(&mut model, &cached, precision)?;
                }
                cached => {
                    load_decision_pytorch(&mut model, path)?;
                    let temporary = cached.is_none();
                    let copy = cached.unwrap_or_else(|| {
                        std::env::temp_dir()
                            .join(format!("rverdict-{}.safetensors", std::process::id()))
                    });
                    save_converted(&model, &copy)?;
                    if precision != Precision::F32 {
                        model = checkpoint.config.init_decision_model::<Dispatch>(&device);
                        load_decision_safetensors(&mut model, &copy, precision)?;
                    }
                    if temporary {
                        let _ = std::fs::remove_file(&copy);
                    }
                }
            },
        }
        Ok(Self {
            model,
            device,
            backend: backend.name,
            packer: Packer::new(
                &checkpoint.tokenizer,
                checkpoint.settings.digit_split,
                checkpoint.config.max_position_embeddings,
            )?,
            config: checkpoint.config.clone(),
            settings: checkpoint.settings.clone(),
            name: checkpoint.name.clone(),
        })
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn decide(&self, request: &Request) -> Result<Response, EngineError> {
        let questions = request.parse_questions()?;
        let state = state_text(&request.state);
        let debias = self.settings.calibration.noul_prior.is_some();

        let mut rows = Vec::new();
        let mut planned = Vec::with_capacity(questions.len());
        for (id, question) in questions {
            let rendered = render(&question);
            let (fitted, cut) =
                self.packer
                    .fit_state(&state, &rendered.instructions, &rendered.options)?;
            rows.push(
                self.packer
                    .pack(&fitted, &rendered.instructions, &rendered.options)?,
            );
            let row = rows.len() - 1;
            let null_row = if debias && rendered.kind == (RenderedKind::Noul { explicit: false }) {
                rows.push(
                    self.packer
                        .pack("", &rendered.instructions, &rendered.options)?,
                );
                Some(rows.len() - 1)
            } else {
                None
            };
            planned.push(Planned {
                state_tokens: self.packer.count(&fitted)?,
                id,
                rendered,
                row,
                null_row,
                cut,
            });
        }

        let logits = self.logits(&rows);
        let calibration = &self.settings.calibration;
        let answers: OrderedMap<Answer> = planned
            .iter()
            .map(|p| {
                let null = p.null_row.map(|r| logits[r].as_slice());
                let answer = calibration.answer(&p.rendered, &logits[p.row], null, p.state_tokens);
                (p.id.clone(), answer)
            })
            .collect();

        let cuts: Vec<Cut> = planned.iter().filter_map(|p| p.cut).collect();
        let truncation = cuts
            .iter()
            .max_by_key(|c| c.state_tokens)
            .map(|worst| Truncation {
                state_tokens: worst.state_tokens,
                kept_tokens: worst.kept_tokens,
                strategy: "middle".into(),
                questions_affected: cuts.len(),
            });

        Ok(Response {
            model: self.name.clone(),
            usage: Usage {
                input_tokens: rows.iter().map(|r| r.token_ids.len()).sum(),
                output_tokens: answers.len(),
            },
            answers,
            truncation,
        })
    }

    /// Option logits per row, in row order. Rows are sorted by length and
    /// grouped so each forward pass pads as little as possible.
    fn logits(&self, rows: &[PackedSequence]) -> Vec<Vec<f32>> {
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].token_ids.len());

        let mut out = vec![Vec::new(); rows.len()];
        let mut start = 0;
        while start < order.len() {
            let mut end = start + 1;
            while end < order.len() {
                let width = rows[order[end]].token_ids.len();
                let count = end - start + 1;
                if count * width > BATCH_TOKENS || count * width * width > BATCH_MASK_CELLS {
                    break;
                }
                end += 1;
            }
            let batch: Vec<PackedSequence> =
                order[start..end].iter().map(|&i| rows[i].clone()).collect();
            for (&row, logits) in order[start..end].iter().zip(self.forward(&batch)) {
                out[row] = logits;
            }
            start = end;
        }
        out
    }

    fn forward(&self, batch: &[PackedSequence]) -> Vec<Vec<f32>> {
        let attention = if self.settings.independent_options {
            OptionAttention::Independent
        } else {
            OptionAttention::Shared
        };
        let input = build_input::<Dispatch>(
            batch,
            attention,
            self.config.pad_token_id,
            self.config.sliding_window,
            &self.device,
        );
        let markers: Vec<Vec<usize>> = batch.iter().map(|s| s.markers.clone()).collect();
        let flat: Vec<f32> = self
            .model
            .option_logits(input, &markers)
            .into_data()
            .to_vec()
            .expect("option logits are f32");
        let mut flat = flat.into_iter();
        markers
            .iter()
            .map(|m| flat.by_ref().take(m.len()).collect())
            .collect()
    }
}
