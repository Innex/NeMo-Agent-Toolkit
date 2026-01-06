# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.common import OptionalSecretStr
from nat.data_models.common import SerializableSecretStr
from nat.data_models.common import get_secret_value
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.mixin.batch_config_mixin import BatchConfigMixin
from nat.observability.mixin.collector_config_mixin import CollectorConfigMixin

logger = logging.getLogger(__name__)


class LangfuseTelemetryExporter(BatchConfigMixin, TelemetryExporterBaseConfig, name="langfuse"):
    """A telemetry exporter to transmit traces to externally hosted langfuse service."""

    endpoint: str = Field(description="The langfuse OTEL endpoint (/api/public/otel/v1/traces)")
    public_key: SerializableSecretStr = Field(description="The Langfuse public key",
                                              default_factory=lambda: SerializableSecretStr(""))
    secret_key: SerializableSecretStr = Field(description="The Langfuse secret key",
                                              default_factory=lambda: SerializableSecretStr(""))
    resource_attributes: dict[str, str] = Field(default_factory=dict,
                                                description="The resource attributes to add to the span")


@register_telemetry_exporter(config_type=LangfuseTelemetryExporter)
async def langfuse_telemetry_exporter(config: LangfuseTelemetryExporter, builder: Builder):

    import base64

    from nat.plugins.opentelemetry import OTLPSpanAdapterExporter

    secret_key = get_secret_value(config.secret_key) if config.secret_key else os.environ.get("LANGFUSE_SECRET_KEY")
    public_key = get_secret_value(config.public_key) if config.public_key else os.environ.get("LANGFUSE_PUBLIC_KEY")
    if not secret_key or not public_key:
        raise ValueError("secret and public keys are required for langfuse")

    credentials = f"{public_key}:{secret_key}".encode()
    auth_header = base64.b64encode(credentials).decode("utf-8")
    headers = {"Authorization": f"Basic {auth_header}"}

    yield OTLPSpanAdapterExporter(endpoint=config.endpoint,
                                  headers=headers,
                                  batch_size=config.batch_size,
                                  flush_interval=config.flush_interval,
                                  max_queue_size=config.max_queue_size,
                                  drop_on_overflow=config.drop_on_overflow,
                                  shutdown_timeout=config.shutdown_timeout)


class LangsmithTelemetryExporter(BatchConfigMixin, CollectorConfigMixin, TelemetryExporterBaseConfig, name="langsmith"):
    """A telemetry exporter to transmit traces to externally hosted langsmith service."""

    endpoint: str = Field(
        description="The langsmith OTEL endpoint",
        default="https://api.smith.langchain.com/otel/v1/traces",
    )
    api_key: SerializableSecretStr = Field(description="The Langsmith API key",
                                           default_factory=lambda: SerializableSecretStr(""))
    resource_attributes: dict[str, str] = Field(default_factory=dict,
                                                description="The resource attributes to add to the span")


@register_telemetry_exporter(config_type=LangsmithTelemetryExporter)
async def langsmith_telemetry_exporter(config: LangsmithTelemetryExporter, builder: Builder):
    """Create a Langsmith telemetry exporter."""

    from nat.plugins.opentelemetry import OTLPSpanAdapterExporter

    api_key = get_secret_value(config.api_key) if config.api_key else os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        raise ValueError("API key is required for langsmith")

    headers = {"x-api-key": api_key, "Langsmith-Project": config.project}
    yield OTLPSpanAdapterExporter(endpoint=config.endpoint,
                                  headers=headers,
                                  batch_size=config.batch_size,
                                  flush_interval=config.flush_interval,
                                  max_queue_size=config.max_queue_size,
                                  drop_on_overflow=config.drop_on_overflow,
                                  shutdown_timeout=config.shutdown_timeout)


class OtelCollectorTelemetryExporter(BatchConfigMixin,
                                     CollectorConfigMixin,
                                     TelemetryExporterBaseConfig,
                                     name="otelcollector"):
    """A telemetry exporter to transmit traces to externally hosted otel collector service."""

    resource_attributes: dict[str, str] = Field(default_factory=dict,
                                                description="The resource attributes to add to the span")


@register_telemetry_exporter(config_type=OtelCollectorTelemetryExporter)
async def otel_telemetry_exporter(config: OtelCollectorTelemetryExporter, builder: Builder):
    """Create an OpenTelemetry telemetry exporter."""

    from nat.plugins.opentelemetry import OTLPSpanAdapterExporter
    from nat.plugins.opentelemetry.otel_span_exporter import get_opentelemetry_sdk_version

    # Default resource attributes
    default_resource_attributes = {
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.version": get_opentelemetry_sdk_version(),
        "service.name": config.project,
    }

    # Merge defaults with config, giving precedence to config
    merged_resource_attributes = {**default_resource_attributes, **config.resource_attributes}

    yield OTLPSpanAdapterExporter(endpoint=config.endpoint,
                                  resource_attributes=merged_resource_attributes,
                                  batch_size=config.batch_size,
                                  flush_interval=config.flush_interval,
                                  max_queue_size=config.max_queue_size,
                                  drop_on_overflow=config.drop_on_overflow,
                                  shutdown_timeout=config.shutdown_timeout)


class PatronusTelemetryExporter(BatchConfigMixin, CollectorConfigMixin, TelemetryExporterBaseConfig, name="patronus"):
    """A telemetry exporter to transmit traces to Patronus service."""

    api_key: SerializableSecretStr = Field(description="The Patronus API key",
                                           default_factory=lambda: SerializableSecretStr(""))
    resource_attributes: dict[str, str] = Field(default_factory=dict,
                                                description="The resource attributes to add to the span")


@register_telemetry_exporter(config_type=PatronusTelemetryExporter)
async def patronus_telemetry_exporter(config: PatronusTelemetryExporter, builder: Builder):
    """Create a Patronus telemetry exporter."""

    from nat.plugins.opentelemetry import OTLPSpanAdapterExporter

    api_key = get_secret_value(config.api_key) if config.api_key else os.environ.get("PATRONUS_API_KEY")
    if not api_key:
        raise ValueError("API key is required for Patronus")

    headers = {
        "x-api-key": api_key,
        "pat-project-name": config.project,
    }
    yield OTLPSpanAdapterExporter(endpoint=config.endpoint,
                                  headers=headers,
                                  batch_size=config.batch_size,
                                  flush_interval=config.flush_interval,
                                  max_queue_size=config.max_queue_size,
                                  drop_on_overflow=config.drop_on_overflow,
                                  shutdown_timeout=config.shutdown_timeout,
                                  protocol="grpc")


class GalileoTelemetryExporter(BatchConfigMixin, CollectorConfigMixin, TelemetryExporterBaseConfig, name="galileo"):
    """A telemetry exporter to transmit traces to externally hosted galileo service."""

    endpoint: str = Field(description="The galileo endpoint to export telemetry traces.",
                          default="https://app.galileo.ai/api/galileo/otel/traces")
    logstream: str = Field(description="The logstream name to group the telemetry traces.")
    api_key: SerializableSecretStr = Field(description="The api key to authenticate with the galileo service.")


@register_telemetry_exporter(config_type=GalileoTelemetryExporter)
async def galileo_telemetry_exporter(config: GalileoTelemetryExporter, builder: Builder):
    """Create a Galileo telemetry exporter."""

    from nat.plugins.opentelemetry import OTLPSpanAdapterExporter

    headers = {
        "Galileo-API-Key": get_secret_value(config.api_key),
        "logstream": config.logstream,
        "project": config.project,
    }

    yield OTLPSpanAdapterExporter(
        endpoint=config.endpoint,
        headers=headers,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
    )


class DBNLTelemetryExporter(BatchConfigMixin, TelemetryExporterBaseConfig, name="dbnl"):
    """A telemetry exporter to transmit traces to DBNL."""

    api_url: str | None = Field(description="The DBNL API URL.", default=None)
    api_token: OptionalSecretStr = Field(description="The DBNL API token.", default=None)
    project_id: str | None = Field(description="The DBNL project id.", default=None)


@register_telemetry_exporter(config_type=DBNLTelemetryExporter)
async def dbnl_telemetry_exporter(config: DBNLTelemetryExporter, builder: Builder):
    """Create a DBNL telemetry exporter."""

    from nat.plugins.opentelemetry import OTLPSpanAdapterExporter

    api_token = get_secret_value(config.api_token) if config.api_token else os.environ.get("DBNL_API_TOKEN")
    if not api_token:
        raise ValueError("API token is required for DBNL")
    project_id = config.project_id or os.environ.get("DBNL_PROJECT_ID")
    if not project_id:
        raise ValueError("Project id is required for DBNL")

    headers = {
        "Authorization": f"Bearer {api_token}",
        "x-dbnl-project-id": project_id,
    }

    api_url = config.api_url or os.environ.get("DBNL_API_URL")
    if not api_url:
        raise ValueError("API url is required for DBNL")
    endpoint = api_url.rstrip("/") + "/otel/v1/traces"

    yield OTLPSpanAdapterExporter(
        endpoint=endpoint,
        headers=headers,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
    )


class BraintrustTelemetryExporter(BatchConfigMixin, CollectorConfigMixin, TelemetryExporterBaseConfig, name="braintrust"):
    """A telemetry exporter to transmit traces to Braintrust for AI observability and evaluation."""

    endpoint: str = Field(
        description="The Braintrust OTEL endpoint",
        default="https://api.braintrust.dev/otel/v1/traces",
    )
    api_key: SerializableSecretStr = Field(description="The Braintrust API key",
                                           default_factory=lambda: SerializableSecretStr(""))
    resource_attributes: dict[str, str] = Field(default_factory=dict,
                                                description="The resource attributes to add to the span")


# Attribute mappings from OpenInference to Braintrust GenAI semantic conventions
BRAINTRUST_ATTRIBUTE_MAPPINGS = {
    "input.value": "gen_ai.prompt",
    "output.value": "gen_ai.completion",
    "llm.token_count.prompt": "gen_ai.usage.prompt_tokens",
    "llm.token_count.completion": "gen_ai.usage.completion_tokens",
    "llm.token_count.total": "gen_ai.usage.total_tokens",
    "llm.model_name": "gen_ai.request.model",
}

# Attributes to remove after mapping (redundant - captured elsewhere in Braintrust schema)
# These are removed to reduce metadata clutter while preserving all information:
# - input.value/output.value -> extracted to Braintrust input/output fields via gen_ai.*
# - MIME types -> not needed for Braintrust display
# - nat.event_timestamp -> captured in metrics.start/metrics.end
# - nat.span.kind/openinference.span.kind -> mapped to span_attributes.type
# - llm.token_count.* -> mapped to gen_ai.usage.*
# - nat.metadata.mime_type -> not needed if nat.metadata exists
BRAINTRUST_REDUNDANT_ATTRIBUTES = {
    "input.value",
    "output.value",
    "input.mime_type",
    "output.mime_type",
    "nat.event_timestamp",
    "nat.span.kind",
    "openinference.span.kind",
    "llm.token_count.prompt",
    "llm.token_count.completion",
    "llm.token_count.total",
    "llm.model_name",
    "nat.metadata.mime_type",
}

# Map OpenInference span kinds to Braintrust span types
# See: https://www.braintrust.dev/docs/reference/span-types
OPENINFERENCE_TO_BRAINTRUST_TYPE = {
    "LLM": "llm",
    "CHAIN": "task",
    "TOOL": "tool",
    "AGENT": "task",
    "EMBEDDING": "task",
    "RETRIEVER": "task",
    "RERANKER": "task",
    "GUARDRAIL": "task",
    "EVALUATOR": "score",
    "UNKNOWN": "task",
}


def _get_improved_span_name(span_name: str, attrs: dict) -> str:
    """Generate an improved span name for better display in Braintrust UI.

    Args:
        span_name: The original span name.
        attrs: The span attributes dictionary.

    Returns:
        An improved span name for display.
    """
    # Handle the generic <workflow> name
    if span_name == "<workflow>":
        # Try to get a more descriptive name from attributes
        function_name = attrs.get("nat.function.name")
        if function_name and function_name != "<workflow>" and function_name != "root":
            return function_name

        # Use event type to create a descriptive name
        event_type = attrs.get("nat.event_type", "")
        if "WORKFLOW" in event_type:
            return "Workflow"
        elif "FUNCTION" in event_type:
            return "Function"
        elif "AGENT" in event_type:
            return "Agent"

        # Fall back to span kind if available
        span_kind = attrs.get("nat.span.kind") or attrs.get("openinference.span.kind")
        if span_kind:
            return span_kind.replace("_", " ").title()

        return "Workflow"

    return span_name


def _transform_span_attributes_for_braintrust(span) -> None:
    """Transform span attributes from OpenInference to Braintrust GenAI conventions.

    This modifies the span's attributes in-place to map OpenInference semantic
    conventions to Braintrust's expected GenAI semantic conventions, including
    proper span type classification and improved span naming.

    Args:
        span: The OtelSpan to transform.
    """
    if not hasattr(span, '_attributes') or span._attributes is None:
        return

    attrs = span._attributes

    # Improve span name for better display in Braintrust UI
    if hasattr(span, '_name') and span._name:
        span._name = _get_improved_span_name(span._name, attrs)

    # Map OpenInference attribute names to Braintrust GenAI conventions
    for old_key, new_key in BRAINTRUST_ATTRIBUTE_MAPPINGS.items():
        if old_key in attrs:
            attrs[new_key] = attrs[old_key]

    # Map OpenInference span kind to Braintrust span type
    # This ensures proper categorization of spans (llm, tool, task, etc.)
    openinference_kind = attrs.get("openinference.span.kind")
    if openinference_kind:
        bt_type = OPENINFERENCE_TO_BRAINTRUST_TYPE.get(openinference_kind, "task")
        attrs["braintrust.span_attributes"] = json.dumps({"type": bt_type})

    # Remove redundant attributes to reduce metadata clutter
    # These are captured elsewhere in Braintrust schema (input/output fields, metrics, span_attributes)
    for key in BRAINTRUST_REDUNDANT_ATTRIBUTES:
        attrs.pop(key, None)


@register_telemetry_exporter(config_type=BraintrustTelemetryExporter)
async def braintrust_telemetry_exporter(config: BraintrustTelemetryExporter, builder: Builder):
    """Create a Braintrust telemetry exporter."""

    from nat.plugins.opentelemetry import OTLPSpanAdapterExporter
    from nat.plugins.opentelemetry.otel_span import OtelSpan

    api_key = get_secret_value(config.api_key) if config.api_key else os.environ.get("BRAINTRUST_API_KEY")
    if not api_key:
        raise ValueError("API key is required for Braintrust")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-bt-parent": f"project_name:{config.project}",
    }

    class BraintrustOTLPSpanAdapterExporter(OTLPSpanAdapterExporter):

        async def export_otel_spans(self, spans: list[OtelSpan]) -> None:
            for span in spans:
                _transform_span_attributes_for_braintrust(span)
            await super().export_otel_spans(spans)

    yield BraintrustOTLPSpanAdapterExporter(
        endpoint=config.endpoint,
        headers=headers,
        resource_attributes=config.resource_attributes,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
    )
