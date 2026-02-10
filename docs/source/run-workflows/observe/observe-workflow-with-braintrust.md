<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Observing a Workflow with Braintrust

This guide provides a step-by-step process to enable observability in a NeMo Agent Toolkit workflow using Braintrust for tracing. By the end of this guide, you will have:

- Configured telemetry in your workflow.
- Ability to view traces in the Braintrust platform.

## Step 1: Create a Braintrust Account

1. Visit [https://www.braintrust.dev](https://www.braintrust.dev) and sign up for an account.
2. Once logged in, navigate to your organization settings to generate an API key.

## Step 2: Create a Project

Create a new project in Braintrust to organize your traces:

1. Navigate to the Braintrust dashboard.
2. Click on **Projects** in the sidebar.
3. Click **+ New Project**.
4. Name your project (e.g., `nat-calculator`).
5. Note down the **Project Name** for configuration.

## Step 3: Configure Your Environment

Set the following environment variables in your terminal:

```bash
export BRAINTRUST_API_KEY=<your_api_key>
```

Alternatively, you can provide the API key directly in your workflow configuration.

## Step 4: Install the NeMo Agent Toolkit OpenTelemetry Subpackages

```bash
# Install specific telemetry extras required for Braintrust
uv pip install -e '.[opentelemetry]'
```

## Step 5: Modify NeMo Agent Toolkit Workflow Configuration

Update your workflow configuration file to include the telemetry settings.

Example configuration:
```yaml
general:
  telemetry:
    tracing:
      braintrust:
        _type: braintrust
        project: nat-calculator
        # Optional: Override the default endpoint for self-hosted deployments
        # endpoint: https://api.braintrust.dev/otel/v1/traces
```

You can also specify the API key directly in the configuration:
```yaml
general:
  telemetry:
    tracing:
      braintrust:
        _type: braintrust
        project: nat-calculator
        api_key: ${BRAINTRUST_API_KEY}
```

## Step 6: Run the workflow

From the root directory of the NeMo Agent Toolkit library, install dependencies and run the pre-configured `simple_calculator_observability` example.

**Example:**

```bash
# Install the workflow and plugins
uv pip install -e examples/observability/simple_calculator_observability/

# Run the workflow with Braintrust telemetry settings
nat run --config_file examples/observability/simple_calculator_observability/configs/config-braintrust.yml --input "What is 1*2?"
```

As the workflow runs, telemetry data will start showing up in Braintrust.

## Step 7: Analyze Traces Data in Braintrust

Analyze the traces in Braintrust:

1. Navigate to [https://www.braintrust.dev](https://www.braintrust.dev) and log in.
2. Go to **Projects** > **nat-calculator** (or your project name).
3. Click on **Logs** to view your traces.
4. Select any trace to view detailed span information, inputs, outputs, and timing data.

```{figure} /_static/braintrust-trace.png
:alt: Braintrust Trace View
:align: center

Example trace view in Braintrust showing workflow spans, inputs, outputs, and timing data.
```

## Additional Features

Braintrust provides additional observability features beyond basic tracing:

- **Evaluation**: Run automated evaluations on your AI outputs with built-in and custom scorers.
- **Experiments**: Compare different model configurations and prompt variations.
- **Datasets**: Curate golden datasets from your production traces.
- **Prompt Management**: Version and deploy prompts with A/B testing capabilities.
- **Human Review**: Set up review queues for team-based quality analysis.

For additional help, see the [Braintrust documentation](https://www.braintrust.dev/docs).
