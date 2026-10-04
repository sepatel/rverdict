# Phase 4: Post Office shadow mode, first backfill

Date: 2026-10-04. Shadow mode is built (plan section 6.7) and has been run
headless over a copy of the production Post Office database: 2,602 messages
from two accounts, 20 Sep to 4 Oct, of which **2,048 rule decisions were made
by the LLM**. Von 1.2 zero-shot, Von's own calibration, f16 on the reference
890M, states capped at 2,048 tokens. The live two-week run has not started:
local decisions are off by default and are turned on in the app.

## Agreement with the LLM, per rule

| Rule (account) | Asked as | n | Agree | LLM's usual answer | Agree on the rest | ≥ 0.9 confident | Held-out NLL, refit |
|---|---|---|---|---|---|---|---|
| Saarth's Activities (sep) | menu, 1 label | 664 | **98%** | 98% | 31% of 13 | 0% | 0.40 → **0.09** |
| School Emails (sep) | menu, 5 labels | 49 | **78%** | 84% | 0% of 8 | 0% | 1.20 → 1.00 |
| Mongo Alerts (sej) | binary | 13 | 77% | 69% | 100% of 4 | 0% | – |
| Mongo Alerts (sej) | noul | 13 | 69% | 69% | 75% of 4 | 0% | – |
| Simple Classification (sej) | menu, 4 labels | 301 | 21% | 35% | 31% of 196 | 0% | 1.66 → 1.59 |
| Simple Classification (sep) | menu, 4 labels | 576 | 15% | 58% | 23% of 241 | 0% | 1.57 → 1.57 |
| Auto Classification (sej) | menu, 61 labels | 267 | 1% | 13% | 1% of 232 | 0% | 3.95 → 3.93 |
| Auto Classification (sep) | menu, 106 labels | 140 | 1% | 20% | 1% of 112 | 0% | 3.94 → 3.36 |
| Trash GitHub Notifications (both, older rule versions) | binary / noul | 31 | 29–79% / 0–25% | 100% | – | 0% | – |

- **Zero-shot Von is not ready to act on any rule**: nothing reaches 0.9
  confidence. That is the honest outcome, and the bar works: no rule has a
  single confident disagreement.
- **Where the rule describes its options, Von follows the LLM**; where it
  does not, it does not. The menus that agree best are the ones with
  specific descriptions. Choosing from 61–106 bare label names fails
  completely, and "None of these" wins most of those.
- **Per-rule calibration refits help most where Von already ranks
  correctly** (Saarth: NLL 0.40 → 0.09). They cannot fix a wrong ranking,
  which is what training (Phase 3) and personal fine-tuning (Phase 5) are for.
- **Two framings for instruction-only rules**: binary agreed more than noul
  on both rules that had them (77% vs 69%, 79% vs 25%), but on 13 and 24
  decisions. Too few to drop either yet.

## Bugs found by the backfill, all fixed

1. **Von's temperature map extrapolated to large menus.** Its `n_options`
   term (−3.56 per 8 options) was fitted on JevBench, which has 2–6 options.
   At 62–107 options it drove the temperature to its 0.3 floor: Auto
   Classification answered "None of these" 327 times at 0.96 average
   confidence while the LLM almost always chose a label, and 310 of its 312
   confident answers disagreed. The map now uses at most the option count it
   was fitted on (`max_options`, 6 for Von, recorded by refits); confidence
   on those questions dropped to about 0.05, an honest "don't know". Any
   large menu would have hit this.
2. **Bare option names.** Rule prompts describe each label (`- "Cold
   Outreach" -- feels like a promotional…`) and what NO_MATCH means; the
   mapping now uses those descriptions. Saarth's Activities: 0% → 98%
   agreement on the same emails.
3. **Sub-labels took their parent's description** (`Education/High` matched
   `Education` as a prefix). School Emails: 4% → 78%.

## Speed

Per decision, 890M, including long emails: median 0.1–1.2 s in f16
depending on the rule (menus with long prompts and long emails are the
slowest); 2.5 s average in f32 on the long-email rules. The full backfill
took 38 minutes in f16. Live volume is 150–300 decisions a day, so even f32
on a slower GPU keeps up easily.

## Safety checks

- Migration 027 on a copy of the production database; a
  `post-office.db.pre-027.bak` copy is written first. Today's build (main at
  `7d2a58c`) opens the migrated database, runs its own migrations (nothing
  to do) and reads workflow data normally.
- The live decision path is unchanged. The only live change is in history
  ingestion: label changes are kept for feedback, only while local decisions
  are enabled, and a failure there is logged and ignored.
- The service loop: off does nothing; on loads the model in about 5 s and
  shadows; off again finishes the batch and unloads the model.

## Radeon 680M (the production machine)

Backend checklist (`scripts/backend-check.sh`), Mesa 25.2.8 RADV, all
passed:

| Check | Result |
|---|---|
| JevBench, Vulkan f32 vs CPU | 137/231 both, 0 predictions differ, max Δp 0.0001, p50 227 ms |
| JevBench, Vulkan f16 vs CPU | 0 predictions differ, max Δp 0.0148, p50 136 ms |
| JevBench, options reversed | 0 predictions differ |
| 24 longest Enron emails (2,048 tokens), Vulkan f32 vs CPU | max logit Δ 0.0005, 0 predictions differ |

f16 is 1.7× faster there (3× on the 890M). It matches on JevBench; the
long-email comparison ran in f32 only, so f16 stays opt-in on that machine
until a long-email f16 comparison passes too.
