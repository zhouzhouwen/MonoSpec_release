# Data

## Source

All experiments come from the public pool-boiling dataset of H. Pandey and H. Hu (University of Arkansas), released under the CC0 1.0 license on Harvard Dataverse and described in the article of H. Pandey, C. Li, and H. Hu:

- Dataset: Pool Boiling Dataset: Synchronized Acoustic, Optical, and Thermal Measurements for Steady-State and Transient Heat Loads, [doi:10.7910/DVN/6GLGC6](https://doi.org/10.7910/DVN/6GLGC6)
- Article: H. Pandey, C. Li, H. Hu, Multimodal boiling dataset with synchronized acoustic, optical, and thermal measurements under steady-state and transient heat loads, Data in Brief 55 (2024) 110582, [doi:10.1016/j.dib.2024.110582](https://doi.org/10.1016/j.dib.2024.110582)

The raw recordings (hydrophone, acoustic-emission, thermocouple, and video files, about 65 GB) are not copied into this repository. Download them from the dataset page. MonoSpec uses the hydrophone (`*_Hydrophone_*.lvm`) and thermocouple (`*_Temperature_*.lvm`) files of 65 experiments (about 28 GB) on five heating surfaces (55 steady runs, 5 step records, 5 transient records).

## Feature file

`scripts/train.py prepare` converts the raw files into one compact feature file (about 80 MB, `data/features.npz`), which `scripts/train.py` and `scripts/test.py` read:

```bash
python scripts/train.py prepare --raw <folder with the Dataverse files> --out data/features.npz
```

The file is a compressed NumPy archive (load with `allow_pickle=False`). Every experiment has the entries below under the prefix `<key>__`.

| Entry | Content |
|---|---|
| `index` | 0-d text array holding a JSON list (`json.loads(str(z["index"]))`), one record per experiment: `key`, `surface`, `category` (`steady`, `step_chf`, `ramp_chf` = transient record), `role` (`steady`, `top_steady` = highest steady run of the surface, `step_chf`, `ramp_chf`), `nominal_power` (power level of the file name, steady runs; null for CHF records; not the measured heat flux), `duration_s`, `t_exc` (crisis onset in s; null for steady runs), `q_mean` (mean heat flux, W/cm²), `q_event_max` (peak heat flux of a CHF record, W/cm²; null for steady runs), `hydro_native_hz` (native hydrophone rate) |
| `<key>__logmel` | float32 `[32, K]`, log-mel band levels (dB) between 20 and 460 Hz, 16 frames per second |
| `<key>__level` | float32 `[K]`, frame level (dB) of the 20–460 Hz band |
| `<key>__power` | float64 `[K]`, linear band power of every frame in squared units of the hydrophone channel; `level` = 10 log10(`power` + 1e-10) |
| `<key>__hi` | float32 `[16, K]`, log-mel band levels (dB) between 460 and 3680 Hz, NaN where a band is not available; absent when no band is available (Polished MC) |
| `<key>__hi_avail` | bool `[16]`, availability of every high band; absent together with `hi` |
| `<key>__q` | float32 `[n]`, heat flux (W/cm²) from the thermocouples at 10 Hz |
| `<key>__r2` | float32 `[n]`, R² of the linear fit of the four thermocouple temperatures at 10 Hz |

Frame k ends at 0.25 + k/16 s, and label sample i is at i/10 s. The window level is 10 log10 of the mean `power` over the 160 frames of a 10 s window. Over the windows of the public dataset it ranges from 5.367 to 26.055 dB, the level range to which the `status` column of `test.py predict` refers.

The index values `duration_s`, `t_exc`, `q_mean`, and `q_event_max`, like the thermocouple column order of every recording, are tabulated constants of the preparation script.

Experiment keys follow the dataset's file names, for example `SS_PH0_105` (steady run of surface PH0 at power level 105, mean heat flux 91.3 W/cm²), `SS_PH0_CHF` (step record), and `TR_PH0_CHF` (transient record).

## Sample

`data/sample.npz` (6 MB) is a small excerpt in the same format for trying the scripts without the raw data: the first 90 s of 20 steady runs (four per surface, including the power levels 15 and 30 and the highest steady run of every surface) and one complete transient record (`TR_PH0_CHF`).

```bash
python scripts/test.py --data data/sample.npz
```

Every window of the sample is identical to the same window of the full file and receives the same estimate; the printed metrics refer to the excerpt, not to the full evaluation. Experiments that are not part of the sample are skipped.
