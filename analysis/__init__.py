"""eeg-analysis: reusable EEG/GSR/resp analysis for stuttering HV/speech data.

Public modules:
  dataset  - EDF discovery + experimental/control group parsing + stage overrides
  eegkit   - general signal helpers (auto dead-channel detection, stages, bands)
"""
from . import dataset, eegkit
from .dataset import discover, parse_name, resolve_dir, summarize, STAGE_OVERRIDES
from .eegkit import (load, eeg_channels, aux_channels, static_dead, dynamic_spans,
                     trim_time, live_channels, channel_status, get, band_pow, bp, iaf, resp_channel,
                     blink_rate, resp_phase_power, stages_from_markers, estimate_stages,
                     get_stages, speak_listen_mask, REGIONS, BANDS, OCCIP, MUSCLE_HB,
                     FRONTAL_MIDLINE)

__all__ = ["dataset", "eegkit"]
