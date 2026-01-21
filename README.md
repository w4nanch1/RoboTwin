# RoboTwin

## Installation

Please follow the official installation tutorial: https://robotwin-platform.github.io/doc/usage/configurations.html#3-notes

## Supported Tasks

Currently supported task: **move_can_pot**

## Checkpoint Setup

Download checkpoints from Google Drive and place them in the `policy/pi05/checkpoints/` directory.

- **Aloha Checkpoint**: Download from [Google Drive](https://drive.google.com/file/d/17JRZ49Lpj9Gx61tz0nl3k7Ulzkz8saVv/view?usp=drive_link)
- **Franka Checkpoint**: Download from [Google Drive]

### Directory Structure

```
policy/pi05/checkpoints/
├── pi05_franka_full_base/
│   └── demo_clean/
│       └── {epoch_id}/
└── pi05_aloha_full_base/
    └── demo_clean/
        └── {epoch_id}/
```

## Evaluation

### Test Aloha Checkpoint

```bash
bash eval.sh move_can_pot demo_clean_aloha pi05_aloha_full_base demo_clean 0 0
```

### Test Franka Checkpoint

```bash
bash eval.sh move_can_pot demo_clean_franka pi05_franka_full_base demo_clean 0 0
```

