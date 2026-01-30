# omramin

Sync **blood pressure** and **weight** measurements from **OMRON connect** to **Garmin Connect**.

`omramin` pulls measurements from your OMRON account and uploads them to Garmin Connect using their respective web APIs. It is intended as a small bridge service, not a full replacement for either app.

> **Note:** For development and testing, consider using a secondary Garmin Connect account.

---

## Features

- Synchronizes:
  - **Weight** measurements from supported OMRON scales
  - **Blood pressure** measurements from supported OMRON blood pressure monitors
- Bridges OMRON connect → Garmin Connect
- Supports multiple devices and users per device (OMRON user slots)

---

## Table of Contents

- [Installation](#installation)
- [Shell Completion](#shell-completion)
- [Updating](#updating)
- [Usage](#usage)
  - [Getting Started](#getting-started)
  - [Configuration](#configuration)
  - [Adding a Device](#adding-a-device)
  - [Synchronizing to Garmin Connect](#synchronizing-to-garmin-connect)
  - [Debugging](#debugging)
- [Commands](#commands)
- [Changelog](CHANGELOG.md)
- [Related Projects](#related-projects)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

**Quick install:**

```sh
# Using uv (recommended)
uv pip install "git+https://github.com/bugficks/omramin.git"

# Using pip
pip install "git+https://github.com/bugficks/omramin.git"

# Run without installing (using uvx)
uvx --from "git+https://github.com/bugficks/omramin.git" omramin init
```

This installs the base package. Most users can start with `omramin init` right away.

### Optional: Bluetooth Device Discovery

If you want to discover OMRON devices via Bluetooth scanning (instead of using API-based discovery or manual MAC entry), install the bluetooth extra:

```sh
# Using uv
uv pip install "git+https://github.com/bugficks/omramin.git[bluetooth]"

# Using pip
pip install "git+https://github.com/bugficks/omramin.git[bluetooth]"
```

**Without Bluetooth support**, you can still:

- Use API-based device discovery: `omramin init --discovery-method api`
- Add devices manually by MAC address: `omramin add --macaddr XX:XX:XX:XX:XX:XX`

### For Developers

Clone and install in editable mode:

```sh
git clone https://github.com/bugficks/omramin.git
cd omramin
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -Ue ".[bluetooth]"
```

---

## Shell Completion

Enable tab completion for command and option names:

**Zsh:**

```sh
# Add to ~/.zshrc
eval "$(_OMRAMIN_COMPLETE=zsh_source omramin)"
```

**Bash:**

```sh
# Add to ~/.bashrc
eval "$(_OMRAMIN_COMPLETE=bash_source omramin)"
```

**Fish:**

```sh
# Add to ~/.config/fish/completions/omramin.fish
_OMRAMIN_COMPLETE=fish_source omramin | source
```

**PowerShell 7+:**

```powershell
# Install completion (run once)
python -m click_pwsh install omramin

# Then restart PowerShell
```

To update completion after upgrading omramin:
```powershell
python -m click_pwsh update omramin
```

After adding the completion script, restart your shell or source the config file:

```sh
source ~/.zshrc   # or ~/.bashrc
```

---

## Updating

```sh
# Using uv
uv pip install --upgrade "git+https://github.com/bugficks/omramin.git"

# Using pip
pip install --upgrade "git+https://github.com/bugficks/omramin.git"

# Editable installations
git pull && pip install -Ue ".[bluetooth]"
```

---

## Usage

### Getting Started

Run the setup wizard:

```sh
omramin init
```

This will:

1. Authenticate with Garmin Connect and OMRON Connect
2. Discover and configure your OMRON devices
3. Create a config file

Then sync measurements:

```sh
omramin sync
```

**Partial setup:**

```sh
omramin init --skip-garmin          # OMRON only
omramin init --skip-devices         # Authentication only
omramin init --discovery-method api # Skip Bluetooth, use API
```

---

### Configuration

**Config file location** (platform-native):

- **Linux:** `~/.config/omramin/config.json`
- **macOS:** `~/Library/Application Support/omramin/config.json`
- **Windows:** `%APPDATA%\omramin\config.json`

**Token storage:**

- **macOS/Linux:** System keyring (Keychain / SecretService)
- **Windows:** File-based (WinVault has size limits)

Override paths with environment variables:

```sh
OMRAMIN_CONFIG=/path/to/config.json omramin sync
```

<details>
<summary><strong>Advanced: Custom Keyring Backends</strong></summary>

Change token storage backend:

```sh
# Use file backend on macOS/Linux
OMRAMIN_KEYRING_BACKEND=file omramin sync

# Use encrypted file backend
OMRAMIN_KEYRING_PASSWORD="secret" omramin sync

# Custom file path
OMRAMIN_KEYRING_FILE=/path/to/tokens.json omramin --keyring-backend file sync
```

Third-party backends (e.g., GPG-encrypted, Pass):

```sh
pip install keyrings.cryptfile
export PYTHON_KEYRING_BACKEND=keyrings.cryptfile.cryptfile.CryptFileKeyring
omramin sync
```

See [keyring backends](https://github.com/jaraco/keyring#third-party-backends) for more options.

</details>

---

### Adding a Device

Three ways to add devices:

**1. Automatic (during init):**

```sh
omramin init  # Discovers devices via API or Bluetooth
```

**2. Interactive add:**

```sh
omramin add  # Scans for Bluetooth devices in pairing mode
```

**3. Manual (if you know the MAC address):**

```sh
omramin add --macaddr 00:11:22:33:44:55 --category scale --name "My Scale"
```

Get MAC addresses from your OMRON account:

```sh
omramin omron list
```

---

### Synchronizing to Garmin Connect

```sh
omramin sync           # Sync today only (default)
omramin sync --days 7  # Sync last 7 days
omramin sync --from 2024-01-01 --to 2024-01-31  # Date range
```

Sync specific devices:

```sh
omramin sync "My Scale"           # By name
omramin sync --category BPM       # All blood pressure monitors
```

Duplicates are automatically detected and skipped.

---

### Debugging

Enable verbose logging:

```sh
omramin --debug sync
```

Targeted debugging:

```sh
OMRAMIN_OMRON_DEBUG=1 omramin sync   # OMRON API only
OMRAMIN_GARMIN_DEBUG=1 omramin sync  # Garmin API only
```

---

## Commands

| Command  | Description                                                       |
| -------- | ----------------------------------------------------------------- |
| `init`   | Interactive setup wizard                                          |
| `sync`   | Sync measurements to Garmin Connect                               |
| `add`    | Add a device                                                      |
| `list`   | List configured devices                                           |
| `config` | Edit device settings                                              |
| `remove` | Remove a device                                                   |
| `garmin` | Garmin Connect authentication (login/logout)                      |
| `omron`  | OMRON Connect operations (login/logout/list/export)                |

### Common Usage

```sh
# Authentication
omramin garmin login
omramin omron login

# Device management
omramin omron list         # Show devices in OMRON account
omramin add                        # Add device interactively
omramin list                       # Show configured devices

# Sync
omramin sync                       # Sync all devices (today only)
omramin sync --days 7              # Sync last 7 days
```

**Help:**

```sh
omramin --help
omramin [command] --help
```

---

## Related Projects

- [`export2garmin`](https://github.com/RobertWojtowicz/export2garmin)  
  Synchronizes data from Mi Body Composition Scale 2 and OMRON blood pressure monitors to Garmin Connect.

---

## Contributing

Contributions are welcome. Bug reports, feature requests, and pull requests help improve:

- Device compatibility
- Error handling and diagnostics

Open an issue or submit a PR on GitHub.

---

## License

This project is distributed under the **GPLv2** license.
