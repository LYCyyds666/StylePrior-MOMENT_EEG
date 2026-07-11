# MOMENT Internal Baselines + Parameter Efficiency Tools

## 0. Put files in project root

```bash
cd /root/autodl-tmp/SageStream_Artifact
unzip moment_internal_and_params_tools.zip
chmod +x *.py *.sh
mkdir -p logs results_moment_internal_parallel paper_tables
```

## 1. Smoke test before full 8-GPU run

Run one APAVA fold for each internal baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python moment_internal_baselines_multidataset.py \
  --dataset APAVA --mode linear --end_fold 1 --epochs 3 \
  --output_dir ./results_moment_internal_smoke

CUDA_VISIBLE_DEVICES=0 python moment_internal_baselines_multidataset.py \
  --dataset APAVA --mode lora --end_fold 1 --epochs 3 \
  --output_dir ./results_moment_internal_smoke

CUDA_VISIBLE_DEVICES=0 python moment_internal_baselines_multidataset.py \
  --dataset APAVA --mode full --end_fold 1 --epochs 3 \
  --output_dir ./results_moment_internal_smoke
```

If one mode fails because `MOMENTPipeline` import fails, send the first error trace.

## 2. 8-GPU run for a dataset and one mode

Example: REFED LoRA.

```bash
nohup bash launch_moment_internal_8gpu.sh REFED lora \
  > logs/launcher_REFED_lora.log 2>&1 &

tail -f logs/launcher_REFED_lora.log
```

Monitor:

```bash
bash monitor_moment_internal.sh REFED lora
```

## 3. Recommended BDCC experiment order

Do not run this while CBraMod is already occupying all 8 GPUs.

After CBraMod jobs finish, run:

```bash
for mode in linear lora full; do
  nohup bash launch_moment_internal_8gpu.sh APAVA $mode \
    > logs/launcher_APAVA_${mode}.log 2>&1 &
  wait
done
```

For SleepEDF and REFED:

```bash
for ds in SleepEDF REFED; do
  for mode in linear lora full; do
    nohup bash launch_moment_internal_8gpu.sh $ds $mode \
      > logs/launcher_${ds}_${mode}.log 2>&1 &
    wait
  done
done
```

If full fine-tuning is too slow, prioritize:
1. linear
2. lora
3. full

## 4. Merge outputs

The launcher automatically merges chunks. Merged results appear under:

```text
results_moment_internal_parallel/<DATASET>/<MODE>/moment_<MODE>_<DATASET>_merged.md
```

## 5. Generate parameter-efficiency table

```bash
python count_trainable_params.py \
  --all_datasets \
  --out ./paper_tables/param_efficiency_all.csv
```

Read:

```bash
cat paper_tables/param_efficiency_all.md
```

## 6. Generate final metric table

After all pkl files are ready:

```bash
python collect_metrics_table.py \
  --roots ./results ./results_cbramod ./results_cbramod_parallel ./results_moment_internal_parallel \
  --out_dir ./paper_tables \
  --debug
```

Then send:

```bash
cat paper_tables/paper_table_core_metrics.md
cat paper_tables/param_efficiency_all.md
```
