# Support Request: `run_command` Sandbox Error on Windows

**Issue Summary**
The AI agent is unable to execute any terminal commands natively on Windows using the `run_command` tool. Attempting to run a command throws a CORTEX-level sandbox configuration error.

**Environment details**
- **Operating System:** Windows
- **Shell:** PowerShell
- **Attempted Action:** Running basic terminal processes/scripts (e.g. `python main.py`, `echo hello`) through the agent's `run_command` action tool.

**Error Message**
The agent infrastructure intercepts the call and drops the execution with the following internal trace:
```text
error executing cascade step: CORTEX_STEP_TYPE_RUN_COMMAND: failed to set up sandbox: sandboxing is not supported on Windows
```

**Steps to Reproduce**
1. Ask the AI agent to run a shell command or script automatically on a Windows machine.
2. The agent correctly drafts the `run_command` tool spec with appropriate `CommandLine`, `Cwd`, etc.
3. The backend execution fails instantaneously with the error above, completely blocking the agent from interacting with the terminal.

**Expected Behavior**
The `run_command` subsystem on a Windows host/shell should either gracefully map the process without sandboxing (if sandboxing is inherently a Linux feature, strictly relying on local environment constraints), or offer a configuration flag to bypass the sandbox for safe local commands.

Please investigate the Cortex agent runtime on Windows hosts, as this totally incapacitates the AI's ability to run, debug, and test code autonomously.
