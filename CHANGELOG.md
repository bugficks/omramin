# Changelog

## Unreleased

### Fixes

- **EU/NA device discovery**: `omron list` and setup device discovery no longer
  return an empty list on the EU/NA v2 API. That API omits the `macAddress`
  field, so every device was dropped by the "skip device without MAC address"
  guard. The MAC is now derived from the BLE local name (e.g.
  `blesmart_…` → `28:FF:B2:10:AA:A4`), then from `deviceSerialID`, matching the
  value produced by Bluetooth discovery and manual `add --macaddr`.

## v0.3.0

- **Garmin auth rewritten**: Migrated to garminconnect>=0.3.0 due to Garmin SSO/OAuth changes; replaces garth-based authentication. Re-login required.

## v0.2.0

### Installation & Setup

- **Simpler installation**: Install directly from GitHub with `pip install "git+https://github.com/..."`
- **Optional Bluetooth**: Bluetooth support is now optional—install only if needed
- **Better setup wizard**: `omramin init` now only saves config after successful completion

### Configuration

- **Platform-native config locations**:
  - Linux: `~/.config/omramin/config.json`
  - macOS: `~/Library/Application Support/omramin/config.json`
  - Windows: `%APPDATA%\omramin\config.json`
- **Secure token storage**: Tokens now stored in system keyring (macOS Keychain, Linux SecretService) for better security

### Synchronization

- **Flexible date ranges**: Use `--from` and `--to` for specific date ranges
- **Filter by category**: Sync only blood pressure monitors or scales with `--category`
- **Regional API support**: Automatic handling of different OMRON API versions worldwide

### Authentication

- **Dedicated login/logout commands**: Separate `garmin login/logout` and `omron login/logout` commands
- **Better credential management**: Improved token handling and refresh

### Device Management

- **List OMRON devices**: New `omramin omron list` command shows all devices in your account
- **Easier device discovery**: Multiple discovery methods (API, Bluetooth, manual MAC entry)
