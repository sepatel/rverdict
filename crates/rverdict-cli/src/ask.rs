use std::io::Read;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Args;
use rverdict_core::Request;
use serde_json::{Map, Value, json};

use crate::model::ModelArgs;

#[derive(Args)]
pub struct Ask {
    #[command(flatten)]
    pub model: ModelArgs,
    /// The text to decide about.
    #[arg(long, conflicts_with_all = ["state_file", "request"])]
    state: Option<String>,
    #[arg(long, conflicts_with = "request")]
    state_file: Option<PathBuf>,
    /// A full JSON request; `-` reads standard input.
    #[arg(long)]
    request: Option<String>,
    /// A yes/no question.
    #[arg(long)]
    noul: Vec<String>,
    /// `"question=option,option,…"`.
    #[arg(long)]
    choice: Vec<String>,
    /// `"question=level,level,…"`, lowest level first.
    #[arg(long)]
    score: Vec<String>,
}

impl Ask {
    pub fn run(&self) -> Result<()> {
        let request = self.request()?;
        let response = self.model.decider()?.decide(&request)?;
        println!("{}", serde_json::to_string_pretty(&response)?);
        Ok(())
    }

    fn request(&self) -> Result<Request> {
        if let Some(source) = &self.request {
            let text = if source == "-" {
                let mut text = String::new();
                std::io::stdin().read_to_string(&mut text)?;
                text
            } else {
                std::fs::read_to_string(source).with_context(|| format!("reading {source}"))?
            };
            return Ok(serde_json::from_str(&text)?);
        }
        let state = match (&self.state, &self.state_file) {
            (Some(text), _) => text.clone(),
            (None, Some(path)) => std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?,
            (None, None) => bail!("give --state, --state-file or --request"),
        };

        let mut questions = Map::new();
        for (i, text) in self.noul.iter().enumerate() {
            questions.insert(
                format!("noul_{i}"),
                json!({"type": "noul", "instructions": text}),
            );
        }
        for (i, spec) in self.choice.iter().enumerate() {
            let (instructions, options) = split_spec(spec)?;
            let criteria: Map<String, Value> =
                options.into_iter().map(|o| (o, Value::Null)).collect();
            questions.insert(
                format!("choice_{i}"),
                json!({"type": "choice", "instructions": instructions, "criteria": criteria}),
            );
        }
        for (i, spec) in self.score.iter().enumerate() {
            let (instructions, levels) = split_spec(spec)?;
            questions.insert(
                format!("score_{i}"),
                json!({"type": "score", "instructions": instructions, "criteria": levels}),
            );
        }
        if questions.is_empty() {
            bail!("ask at least one --noul, --choice or --score question");
        }
        Ok(Request {
            state: Value::String(state),
            model: None,
            questions,
        })
    }
}

fn split_spec(spec: &str) -> Result<(String, Vec<String>)> {
    let (instructions, options) = spec
        .rsplit_once('=')
        .with_context(|| format!("{spec:?} is not \"question=option,option\""))?;
    let options = options
        .split(',')
        .map(|o| o.trim().to_owned())
        .filter(|o| !o.is_empty())
        .collect();
    Ok((instructions.trim().to_owned(), options))
}
