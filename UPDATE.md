# OmniGen Updates

## 2025-01-22 (v1.2.2)
### Improved
- Significant performance improvements in pipeline execution
- Enhanced device management stability
- Optimized memory handling during model operations
- Better error recovery mechanisms

### Fixed
- Pipeline device movement reliability
- Memory management in edge cases
- Error handling during device transitions

## 2024-11-11
### Added
- Preset Prompts Support
  - Added preset prompt selection from data.json
  - Custom prompts take precedence over presets
  - Easy to extend with new presets
  - Default to empty if no preset selected

- Model Precision Selection
  - Added three precision options:
    - Auto: Automatically selects based on VRAM
    - FP16: Full precision (15.5GB VRAM)
    - FP8: Reduced precision (3.4GB VRAM)
  - Auto mode selects FP8 for systems with <8GB VRAM
  - Shows available VRAM in selection message
  - Smart switching between models with proper cleanup

- Memory Management Improvements
  - Three modes available:
    - Balanced (Default): Standard operation mode
    - Speed Priority: Keeps model in VRAM for faster consecutive generations
    - Memory Priority: Aggressive memory saving with model offloading
  - Smart model instance caching
  - Automatic VRAM cleanup when switching models
  - Recommended settings:
    - FP8 model (3.4GB VRAM): Speed Priority mode is safe
    - FP16 model (15.5GB VRAM): Use Memory Priority mode if VRAM limited

- Pipeline Improvements
  - Better device movement handling
  - Original pipeline backup for device movement failures
  - Improved error handling and recovery
  - Component-wise device movement for better stability

### Fixed
- Pipeline device movement issues
- Memory leaks in consecutive generations
- Temporary file cleanup
- Model precision switching issues

### Improved
- Error handling and logging
- VRAM usage monitoring
- Temporary file management with UUID
- Code organization and documentation