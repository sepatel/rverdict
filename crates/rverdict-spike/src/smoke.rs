use anyhow::{Result, anyhow};
use burn::prelude::*;
use rverdict_model::{
    DecisionModel, EncoderConfig, OptionAttention, PackedSequence, build_input,
    load_decision_pytorch,
};

use crate::fetch::ModelFiles;
use crate::parity::diff;
use crate::von::{Calibration, NOUL_FALSE, NOUL_TRUE, Packer, confidence, noul_band, softmax};

pub struct Von<B: Backend> {
    pub model: DecisionModel<B>,
    pub config: EncoderConfig,
    pub packer: Packer,
    pub calibration: Calibration,
    pub device: B::Device,
}

impl<B: Backend> Von<B> {
    pub fn load(device: &B::Device, files: &ModelFiles) -> Result<Self> {
        let config = EncoderConfig::from_file(&files.path("config.json"))?;
        let mut model = config.init_decision_model::<B>(device);
        load_decision_pytorch(&mut model, &files.path("option_marker.pt"))?;
        Ok(Self {
            model,
            packer: Packer::from_file(&files.path("tokenizer.json"))?,
            calibration: Calibration::from_file(&files.path("marker_calibration.json"))?,
            config,
            device: device.clone(),
        })
    }

    pub fn logits(&self, seq: &PackedSequence) -> Result<Vec<f32>> {
        let input = build_input::<B>(
            std::slice::from_ref(seq),
            OptionAttention::Independent,
            self.config.pad_token_id,
            self.config.sliding_window,
            &self.device,
        );
        self.model
            .option_logits(input, std::slice::from_ref(&seq.markers))
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow!("{e:?}"))
    }

    /// Calibrated option probabilities for a `choice` question.
    pub fn choice(
        &self,
        state: &str,
        question: &str,
        options: &[&str],
    ) -> Result<(Vec<f64>, Vec<f32>)> {
        let logits = self.logits(&self.packer.pack(state, question, options)?)?;
        let t = self
            .calibration
            .temperature(&logits, self.packer.state_tokens(state)?);
        Ok((softmax(&logits, t), logits))
    }

    /// P(true) for a `noul` question without criteria: Von's zero-shot
    /// debiasing against the state-free prior, then the band rule.
    pub fn noul(&self, state: &str, question: &str) -> Result<f64> {
        let options = [NOUL_TRUE, NOUL_FALSE];
        let mut logits = self.logits(&self.packer.pack(state, question, &options)?)?;
        let null = self.logits(&self.packer.pack("", question, &options)?)?;
        let (a, b) = self.calibration.noul_prior;
        #[expect(clippy::cast_possible_truncation, reason = "logits are f32")]
        let correction = (a * f64::from(null[0] - null[1]) + b) as f32;
        let raw = logits.clone();
        logits[0] -= correction;
        let t = self
            .calibration
            .temperature(&logits, self.packer.state_tokens(state)?);
        let p = softmax(&logits, t)[0];
        if std::env::var_os("RVERDICT_DEBUG").is_some() {
            eprintln!(
                "  logits {raw:?} null {null:?} correction {correction:.3} T {t:.3} raw P(true) {p:.3}"
            );
        }
        Ok(noul_band(p))
    }
}

const TICKET: &str = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.";
const NEWSLETTER: &str = "From: The Weekly Byte <news@weeklybyte.io>\nSubject: 10 Rust crates you missed this week\n\
Hello reader! Here is this week's roundup of the best new crates. Unsubscribe any time from the link below.";
const RECEIPT: &str = "From: billing@hetzner.com\nSubject: Invoice 2026-0912 for your server\n\
Your monthly invoice of EUR 42.00 for server CX32 has been charged to your card ending 4417.";

pub fn run<B: Backend>(device: &B::Device, files: &ModelFiles) -> Result<()> {
    let von = Von::<B>::load(device, files)?;

    let departments = [
        "billing: invoices, payments, refunds",
        "technical: bugs, outages, system errors",
        "sales: pricing, new contracts",
    ];
    let (probs, logits) = von.choice(
        TICKET,
        "Which department should handle this request?",
        &departments,
    )?;
    println!(
        "choice  department  {} (confidence {:.3})",
        fmt_probs(&["billing", "technical", "sales"], &probs),
        confidence(&probs)
    );

    let mut reversed = departments;
    reversed.reverse();
    let (_, mut reversed_logits) = von.choice(
        TICKET,
        "Which department should handle this request?",
        &reversed,
    )?;
    reversed_logits.reverse();
    println!(
        "        reversed options, logit drift: {}",
        diff(&reversed_logits, &logits)
    );

    let kinds = [
        "a newsletter or marketing email",
        "a receipt, invoice or bill",
        "a personal message from a person",
    ];
    for (name, email) in [("newsletter", NEWSLETTER), ("receipt", RECEIPT)] {
        let (probs, _) = von.choice(email, "What kind of email is this?", &kinds)?;
        println!(
            "choice  {name:<10}  {}",
            fmt_probs(&["newsletter", "receipt", "personal"], &probs)
        );
    }

    for (name, email, question) in [
        (
            "newsletter",
            NEWSLETTER,
            "The email is a newsletter the user subscribed to.",
        ),
        (
            "receipt",
            RECEIPT,
            "The email is a newsletter the user subscribed to.",
        ),
        (
            "receipt",
            RECEIPT,
            "The email confirms a payment was charged.",
        ),
        ("ticket", TICKET, "The customer threatens to cancel."),
    ] {
        println!(
            "noul    {name:<10}  P(true)={:.3}  {question}",
            von.noul(email, question)?
        );
    }
    Ok(())
}

fn fmt_probs(labels: &[&str], probs: &[f64]) -> String {
    labels
        .iter()
        .zip(probs)
        .map(|(label, p)| format!("{label}={p:.3}"))
        .collect::<Vec<_>>()
        .join(" ")
}
