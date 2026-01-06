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

"""Tests for Braintrust telemetry exporter integration."""

import json
import os
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from nat.plugins.opentelemetry.register import (
    BRAINTRUST_ATTRIBUTE_MAPPINGS,
    BRAINTRUST_REDUNDANT_ATTRIBUTES,
    OPENINFERENCE_TO_BRAINTRUST_TYPE,
    BraintrustTelemetryExporter,
    _get_improved_span_name,
    _transform_span_attributes_for_braintrust,
    braintrust_telemetry_exporter,
)


class TestBraintrustAttributeMappings:
    """Test attribute mapping constants."""

    def test_attribute_mappings_exist(self):
        """Test that all expected attribute mappings are defined."""
        expected_mappings = {
            "input.value": "gen_ai.prompt",
            "output.value": "gen_ai.completion",
            "llm.token_count.prompt": "gen_ai.usage.prompt_tokens",
            "llm.token_count.completion": "gen_ai.usage.completion_tokens",
            "llm.token_count.total": "gen_ai.usage.total_tokens",
            "llm.model_name": "gen_ai.request.model",
        }
        assert BRAINTRUST_ATTRIBUTE_MAPPINGS == expected_mappings

    def test_span_type_mappings_exist(self):
        """Test that all expected span type mappings are defined."""
        expected_types = {
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
        assert OPENINFERENCE_TO_BRAINTRUST_TYPE == expected_types

    def test_redundant_attributes_defined(self):
        """Test that redundant attributes to remove are defined."""
        expected_redundant = {
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
        assert BRAINTRUST_REDUNDANT_ATTRIBUTES == expected_redundant


class TestGetImprovedSpanName:
    """Test the _get_improved_span_name function."""

    def test_workflow_span_returns_workflow(self):
        """Test that <workflow> span with WORKFLOW event type returns 'Workflow'."""
        attrs = {"nat.event_type": "WORKFLOW_START"}
        result = _get_improved_span_name("<workflow>", attrs)
        assert result == "Workflow"

    def test_workflow_span_with_function_event_returns_function(self):
        """Test that <workflow> span with FUNCTION event type returns 'Function'."""
        attrs = {"nat.event_type": "FUNCTION_START"}
        result = _get_improved_span_name("<workflow>", attrs)
        assert result == "Function"

    def test_workflow_span_with_agent_event_returns_agent(self):
        """Test that <workflow> span with AGENT event type returns 'Agent'."""
        attrs = {"nat.event_type": "AGENT_START"}
        result = _get_improved_span_name("<workflow>", attrs)
        assert result == "Agent"

    def test_workflow_span_with_function_name_uses_function_name(self):
        """Test that <workflow> span with valid function name uses that name."""
        attrs = {"nat.function.name": "my_custom_function", "nat.event_type": "WORKFLOW_START"}
        result = _get_improved_span_name("<workflow>", attrs)
        assert result == "my_custom_function"

    def test_workflow_span_ignores_root_function_name(self):
        """Test that <workflow> span ignores 'root' function name."""
        attrs = {"nat.function.name": "root", "nat.event_type": "WORKFLOW_START"}
        result = _get_improved_span_name("<workflow>", attrs)
        assert result == "Workflow"

    def test_workflow_span_ignores_workflow_function_name(self):
        """Test that <workflow> span ignores '<workflow>' function name."""
        attrs = {"nat.function.name": "<workflow>", "nat.event_type": "FUNCTION_START"}
        result = _get_improved_span_name("<workflow>", attrs)
        assert result == "Function"

    def test_workflow_span_with_span_kind_fallback(self):
        """Test that <workflow> span falls back to span kind."""
        attrs = {"nat.span.kind": "CUSTOM_TYPE"}
        result = _get_improved_span_name("<workflow>", attrs)
        assert result == "Custom Type"

    def test_workflow_span_with_openinference_kind_fallback(self):
        """Test that <workflow> span falls back to openinference span kind."""
        attrs = {"openinference.span.kind": "RETRIEVER"}
        result = _get_improved_span_name("<workflow>", attrs)
        assert result == "Retriever"

    def test_workflow_span_default_fallback(self):
        """Test that <workflow> span defaults to 'Workflow'."""
        attrs = {}
        result = _get_improved_span_name("<workflow>", attrs)
        assert result == "Workflow"

    def test_non_workflow_span_unchanged(self):
        """Test that non-workflow span names are unchanged."""
        attrs = {"nat.event_type": "LLM_START"}
        result = _get_improved_span_name("gpt-4o-mini", attrs)
        assert result == "gpt-4o-mini"

    def test_tool_span_unchanged(self):
        """Test that tool span names are unchanged."""
        attrs = {"nat.event_type": "TOOL_START"}
        result = _get_improved_span_name("calculator.add", attrs)
        assert result == "calculator.add"


class TestTransformSpanAttributesForBraintrust:
    """Test the _transform_span_attributes_for_braintrust function."""

    def test_transforms_input_value(self):
        """Test that input.value is mapped to gen_ai.prompt and then removed."""
        span = Mock()
        span._name = "test_span"
        span._attributes = {"input.value": "Hello, world!"}

        _transform_span_attributes_for_braintrust(span)

        assert span._attributes["gen_ai.prompt"] == "Hello, world!"
        assert "input.value" not in span._attributes  # Redundant attribute removed

    def test_transforms_output_value(self):
        """Test that output.value is mapped to gen_ai.completion and then removed."""
        span = Mock()
        span._name = "test_span"
        span._attributes = {"output.value": "Response text"}

        _transform_span_attributes_for_braintrust(span)

        assert span._attributes["gen_ai.completion"] == "Response text"
        assert "output.value" not in span._attributes  # Redundant attribute removed

    def test_transforms_token_counts(self):
        """Test that token counts are mapped correctly."""
        span = Mock()
        span._name = "test_span"
        span._attributes = {
            "llm.token_count.prompt": 100,
            "llm.token_count.completion": 50,
            "llm.token_count.total": 150,
        }

        _transform_span_attributes_for_braintrust(span)

        assert span._attributes["gen_ai.usage.prompt_tokens"] == 100
        assert span._attributes["gen_ai.usage.completion_tokens"] == 50
        assert span._attributes["gen_ai.usage.total_tokens"] == 150

    def test_transforms_model_name(self):
        """Test that llm.model_name is mapped to gen_ai.request.model and then removed."""
        span = Mock()
        span._name = "gpt-4o-mini"
        span._attributes = {"llm.model_name": "gpt-4o-mini"}

        _transform_span_attributes_for_braintrust(span)

        assert span._attributes["gen_ai.request.model"] == "gpt-4o-mini"
        assert "llm.model_name" not in span._attributes

    def test_sets_braintrust_span_type_for_llm(self):
        """Test that LLM span kind maps to llm type."""
        span = Mock()
        span._name = "gpt-4o"
        span._attributes = {"openinference.span.kind": "LLM"}

        _transform_span_attributes_for_braintrust(span)

        span_attrs = json.loads(span._attributes["braintrust.span_attributes"])
        assert span_attrs["type"] == "llm"

    def test_sets_braintrust_span_type_for_tool(self):
        """Test that TOOL span kind maps to tool type."""
        span = Mock()
        span._name = "calculator"
        span._attributes = {"openinference.span.kind": "TOOL"}

        _transform_span_attributes_for_braintrust(span)

        span_attrs = json.loads(span._attributes["braintrust.span_attributes"])
        assert span_attrs["type"] == "tool"

    def test_sets_braintrust_span_type_for_chain(self):
        """Test that CHAIN span kind maps to task type."""
        span = Mock()
        span._name = "workflow"
        span._attributes = {"openinference.span.kind": "CHAIN"}

        _transform_span_attributes_for_braintrust(span)

        span_attrs = json.loads(span._attributes["braintrust.span_attributes"])
        assert span_attrs["type"] == "task"

    def test_sets_braintrust_span_type_for_evaluator(self):
        """Test that EVALUATOR span kind maps to score type."""
        span = Mock()
        span._name = "evaluator"
        span._attributes = {"openinference.span.kind": "EVALUATOR"}

        _transform_span_attributes_for_braintrust(span)

        span_attrs = json.loads(span._attributes["braintrust.span_attributes"])
        assert span_attrs["type"] == "score"

    def test_unknown_span_kind_defaults_to_task(self):
        """Test that unknown span kind defaults to task type."""
        span = Mock()
        span._name = "unknown"
        span._attributes = {"openinference.span.kind": "SOME_NEW_TYPE"}

        _transform_span_attributes_for_braintrust(span)

        span_attrs = json.loads(span._attributes["braintrust.span_attributes"])
        assert span_attrs["type"] == "task"

    def test_improves_workflow_span_name(self):
        """Test that <workflow> span name is improved."""
        span = Mock()
        span._name = "<workflow>"
        span._attributes = {"nat.event_type": "WORKFLOW_START"}

        _transform_span_attributes_for_braintrust(span)

        assert span._name == "Workflow"

    def test_removes_all_redundant_attributes(self):
        """Test that all redundant attributes are removed after transformation."""
        span = Mock()
        span._name = "test_span"
        span._attributes = {
            "input.value": "Hello",
            "output.value": "World",
            "input.mime_type": "text/plain",
            "output.mime_type": "text/plain",
            "nat.event_timestamp": 1234567890,
            "nat.span.kind": "WORKFLOW",
            "openinference.span.kind": "CHAIN",
            "llm.token_count.prompt": 100,
            "llm.token_count.completion": 50,
            "llm.token_count.total": 150,
            "nat.metadata.mime_type": "application/json",
            "nat.workflow.run_id": "abc123",  # Should be kept
            "nat.function.name": "my_func",    # Should be kept
        }

        _transform_span_attributes_for_braintrust(span)

        # Verify redundant attributes are removed
        for attr in BRAINTRUST_REDUNDANT_ATTRIBUTES:
            assert attr not in span._attributes, f"{attr} should have been removed"

        # Verify mapped attributes exist
        assert span._attributes["gen_ai.prompt"] == "Hello"
        assert span._attributes["gen_ai.completion"] == "World"
        assert span._attributes["gen_ai.usage.prompt_tokens"] == 100
        assert span._attributes["gen_ai.usage.completion_tokens"] == 50
        assert span._attributes["gen_ai.usage.total_tokens"] == 150

        # Verify non-redundant NAT attributes are preserved
        assert span._attributes["nat.workflow.run_id"] == "abc123"
        assert span._attributes["nat.function.name"] == "my_func"

    def test_handles_missing_attributes(self):
        """Test that function handles span with no attributes."""
        span = Mock()
        span._attributes = None

        # Should not raise
        _transform_span_attributes_for_braintrust(span)

    def test_handles_span_without_attributes_attr(self):
        """Test that function handles span without _attributes."""
        span = object()  # No _attributes attribute

        # Should not raise
        _transform_span_attributes_for_braintrust(span)


class TestBraintrustTelemetryExporterConfig:
    """Test BraintrustTelemetryExporter configuration."""

    def test_default_endpoint(self):
        """Test that default endpoint is set correctly."""
        config = BraintrustTelemetryExporter(project="test-project")
        assert config.endpoint == "https://api.braintrust.dev/otel/v1/traces"

    def test_custom_endpoint(self):
        """Test that custom endpoint can be set."""
        config = BraintrustTelemetryExporter(
            project="test-project",
            endpoint="https://custom.endpoint.com/traces"
        )
        assert config.endpoint == "https://custom.endpoint.com/traces"

    def test_project_required(self):
        """Test that project field is set."""
        config = BraintrustTelemetryExporter(project="my-project")
        assert config.project == "my-project"

    def test_api_key_can_be_set(self):
        """Test that API key can be provided in config."""
        config = BraintrustTelemetryExporter(
            project="test-project",
            api_key="test-api-key"
        )
        assert config.api_key.get_secret_value() == "test-api-key"

    def test_resource_attributes_default_empty(self):
        """Test that resource_attributes defaults to empty dict."""
        config = BraintrustTelemetryExporter(project="test-project")
        assert config.resource_attributes == {}

    def test_resource_attributes_can_be_set(self):
        """Test that resource_attributes can be set."""
        config = BraintrustTelemetryExporter(
            project="test-project",
            resource_attributes={"service.name": "my-service"}
        )
        assert config.resource_attributes == {"service.name": "my-service"}


class TestBraintrustTelemetryExporterFactory:
    """Test the braintrust_telemetry_exporter async factory function."""

    @pytest.fixture
    def mock_builder(self):
        """Create a mock Builder."""
        return Mock()

    @pytest.fixture
    def config_with_key(self):
        """Config with API key set directly."""
        return BraintrustTelemetryExporter(
            project="test-project",
            api_key="test-api-key-123",
        )

    @pytest.fixture
    def config_without_key(self):
        """Config without API key."""
        return BraintrustTelemetryExporter(
            project="test-project",
        )

    @patch("nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporterHTTP")
    async def test_factory_creates_exporter_with_config_api_key(
        self, mock_otlp_http, config_with_key, mock_builder
    ):
        """Test that the factory creates an exporter using the config API key."""
        async with braintrust_telemetry_exporter(config_with_key, mock_builder) as exporter:
            pass

        call_kwargs = mock_otlp_http.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-api-key-123"

    @patch("nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporterHTTP")
    async def test_factory_creates_exporter_with_correct_project_header(
        self, mock_otlp_http, config_with_key, mock_builder
    ):
        """Test that the factory sets the x-bt-parent header with project name."""
        async with braintrust_telemetry_exporter(config_with_key, mock_builder) as exporter:
            pass

        call_kwargs = mock_otlp_http.call_args[1]
        assert call_kwargs["headers"]["x-bt-parent"] == "project_name:test-project"

    @patch.dict(os.environ, {"BRAINTRUST_API_KEY": "env-api-key-456"})
    @patch("nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporterHTTP")
    async def test_factory_falls_back_to_env_var(
        self, mock_otlp_http, config_without_key, mock_builder
    ):
        """Test that the factory falls back to BRAINTRUST_API_KEY env var."""
        async with braintrust_telemetry_exporter(config_without_key, mock_builder) as exporter:
            pass

        call_kwargs = mock_otlp_http.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer env-api-key-456"

    @patch.dict(os.environ, {}, clear=True)
    async def test_factory_raises_without_api_key(self, config_without_key, mock_builder):
        """Test that the factory raises ValueError when no API key is available."""
        with pytest.raises(ValueError, match="API key is required for Braintrust"):
            async with braintrust_telemetry_exporter(config_without_key, mock_builder) as exporter:
                pass

    @patch("nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporterHTTP")
    async def test_factory_passes_batch_config(self, mock_otlp_http, mock_builder):
        """Test that batch configuration is passed through to the exporter."""
        config = BraintrustTelemetryExporter(
            project="test-project",
            api_key="test-key",
            batch_size=100,
            flush_interval=10.0,
            max_queue_size=500,
        )

        async with braintrust_telemetry_exporter(config, mock_builder) as exporter:
            assert exporter is not None
            assert type(exporter).__name__ == "BraintrustOTLPSpanAdapterExporter"

    @patch("nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporterHTTP")
    async def test_factory_passes_endpoint(self, mock_otlp_http, mock_builder):
        """Test that custom endpoint is passed through to the exporter."""
        config = BraintrustTelemetryExporter(
            project="test-project",
            api_key="test-key",
            endpoint="https://custom.braintrust.dev/otel/v1/traces",
        )

        async with braintrust_telemetry_exporter(config, mock_builder) as exporter:
            pass

        call_kwargs = mock_otlp_http.call_args[1]
        assert call_kwargs["endpoint"] == "https://custom.braintrust.dev/otel/v1/traces"

    @patch("nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporterHTTP")
    async def test_factory_yields_braintrust_subclass(self, mock_otlp_http, config_with_key, mock_builder):
        """Test that the factory yields a BraintrustOTLPSpanAdapterExporter subclass."""
        async with braintrust_telemetry_exporter(config_with_key, mock_builder) as exporter:
            assert type(exporter).__name__ == "BraintrustOTLPSpanAdapterExporter"

    @patch("nat.plugins.opentelemetry.mixin.otlp_span_exporter_mixin.OTLPSpanExporterHTTP")
    async def test_subclass_transforms_spans_before_export(self, mock_otlp_http, config_with_key, mock_builder):
        """Test that the subclass transforms span attributes before calling super().export_otel_spans()."""
        async with braintrust_telemetry_exporter(config_with_key, mock_builder) as exporter:
            span = Mock()
            span._name = "<workflow>"
            span._attributes = {
                "input.value": "Hello",
                "openinference.span.kind": "LLM",
                "nat.event_type": "WORKFLOW_START",
            }
            span.set_resource = Mock()

            await exporter.export_otel_spans([span])

            assert span._attributes.get("gen_ai.prompt") == "Hello"
            assert "input.value" not in span._attributes
