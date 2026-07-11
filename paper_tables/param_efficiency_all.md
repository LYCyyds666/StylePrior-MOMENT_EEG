| Dataset | Model | Setting | Total Params | Trainable Params | Trainable Ratio | TTA-updated Params |
|---|---|---|---:|---:|---:|---:|
| APAVA | MOMENT-linear | frozen backbone + trainable head | 35,353,794 | 16,386 | 0.046% |  |
| APAVA | MOMENT-full | full fine-tuning | 35,353,794 | 35,353,794 | 100.000% |  |
| APAVA | MOMENT-lora | LoRA(q,v) + trainable head | 35,468,482 | 131,074 | 0.370% |  |
| APAVA | StylePrior-MOMENT | frozen backbone + trainable SA-MoE + head | 46,067,655 | 10,730,247 | 23.292% | 16,384 |
| APAVA | CBraMod-full | full fine-tuning + classifier | 5,743,970 | 5,743,970 | 100.000% |  |
| SleepEDF | MOMENT-linear | frozen backbone + trainable head | 35,378,373 | 40,965 | 0.116% |  |
| SleepEDF | MOMENT-full | full fine-tuning | 35,378,373 | 35,378,373 | 100.000% |  |
| SleepEDF | MOMENT-lora | LoRA(q,v) + trainable head | 35,493,061 | 155,653 | 0.439% |  |
| SleepEDF | StylePrior-MOMENT | frozen backbone + trainable SA-MoE + head | 46,090,698 | 10,753,290 | 23.331% | 16,384 |
| SleepEDF | CBraMod-full | full fine-tuning + classifier | 5,744,741 | 5,744,741 | 100.000% |  |
| REFED | MOMENT-linear | frozen backbone + trainable head | 35,353,794 | 16,386 | 0.046% |  |
| REFED | MOMENT-full | full fine-tuning | 35,353,794 | 35,353,794 | 100.000% |  |
| REFED | MOMENT-lora | LoRA(q,v) + trainable head | 35,468,482 | 131,074 | 0.370% |  |
| REFED | StylePrior-MOMENT | frozen backbone + trainable SA-MoE + head | 46,072,263 | 10,734,855 | 23.300% | 16,384 |
| REFED | CBraMod-full | full fine-tuning + classifier | 5,743,970 | 5,743,970 | 100.000% |  |
