package codex

import (
	"encoding/json"
	"testing"

	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/stretchr/testify/require"
)

func TestNormalizeResponsesInputInstructions_HoistsSystemRoleStringContent(t *testing.T) {
	request := &dto.OpenAIResponsesRequest{
		Input: json.RawMessage(`[
			{"role":"system","content":"You are a helpful assistant."},
			{"role":"user","content":"hi"}
		]`),
	}

	err := normalizeResponsesInputInstructions(request)
	require.NoError(t, err)

	var instructions string
	require.NoError(t, json.Unmarshal(request.Instructions, &instructions))
	require.Equal(t, "You are a helpful assistant.", instructions)

	var input []map[string]any
	require.NoError(t, json.Unmarshal(request.Input, &input))
	require.Len(t, input, 1)
	require.Equal(t, "user", input[0]["role"])
}

func TestNormalizeResponsesInputInstructions_HoistsDeveloperRole(t *testing.T) {
	// Pi's generic openai-responses provider sends role "developer" (not
	// "system") for reasoning-enabled models, which Codex models are.
	request := &dto.OpenAIResponsesRequest{
		Input: json.RawMessage(`[
			{"role":"developer","content":"You are Pi, a terminal-based coding agent."},
			{"role":"user","content":"hi"}
		]`),
	}

	err := normalizeResponsesInputInstructions(request)
	require.NoError(t, err)

	var instructions string
	require.NoError(t, json.Unmarshal(request.Instructions, &instructions))
	require.Equal(t, "You are Pi, a terminal-based coding agent.", instructions)

	var input []map[string]any
	require.NoError(t, json.Unmarshal(request.Input, &input))
	require.Len(t, input, 1)
	require.Equal(t, "user", input[0]["role"])
}

func TestNormalizeResponsesInputInstructions_HoistsArrayContentParts(t *testing.T) {
	request := &dto.OpenAIResponsesRequest{
		Input: json.RawMessage(`[
			{"role":"developer","content":[{"type":"input_text","text":"line one"},{"type":"input_text","text":"line two"}]},
			{"role":"user","content":"hi"}
		]`),
	}

	err := normalizeResponsesInputInstructions(request)
	require.NoError(t, err)

	var instructions string
	require.NoError(t, json.Unmarshal(request.Instructions, &instructions))
	require.Equal(t, "line one\nline two", instructions)
}

func TestNormalizeResponsesInputInstructions_PreservesNonTextContentParts(t *testing.T) {
	// A system/developer message that also carries an image part cannot be
	// fully represented as plain instruction text; the non-text part must be
	// preserved as a regular input message rather than silently dropped.
	request := &dto.OpenAIResponsesRequest{
		Input: json.RawMessage(`[
			{"role":"developer","content":[{"type":"input_text","text":"look at this"},{"type":"input_image","image_url":"https://example.com/a.png"}]}
		]`),
	}

	err := normalizeResponsesInputInstructions(request)
	require.NoError(t, err)

	var instructions string
	require.NoError(t, json.Unmarshal(request.Instructions, &instructions))
	require.Equal(t, "look at this", instructions)

	var input []map[string]any
	require.NoError(t, json.Unmarshal(request.Input, &input))
	require.Len(t, input, 1)
	require.Equal(t, "user", input[0]["role"])

	content, ok := input[0]["content"].([]any)
	require.True(t, ok)
	require.Len(t, content, 1)
}

func TestNormalizeResponsesInputInstructions_MergesWithExistingInstructions(t *testing.T) {
	request := &dto.OpenAIResponsesRequest{
		Instructions: json.RawMessage(`"top-level instructions"`),
		Input: json.RawMessage(`[
			{"role":"developer","content":"embedded instructions"},
			{"role":"user","content":"hi"}
		]`),
	}

	err := normalizeResponsesInputInstructions(request)
	require.NoError(t, err)

	var instructions string
	require.NoError(t, json.Unmarshal(request.Instructions, &instructions))
	require.Equal(t, "top-level instructions\nembedded instructions", instructions)
}

func TestNormalizeResponsesInputInstructions_NoSystemOrDeveloperMessage_NoOp(t *testing.T) {
	original := json.RawMessage(`[{"role":"user","content":"hi"}]`)
	request := &dto.OpenAIResponsesRequest{
		Input: original,
	}

	err := normalizeResponsesInputInstructions(request)
	require.NoError(t, err)
	require.Nil(t, request.Instructions)
	require.JSONEq(t, string(original), string(request.Input))
}

func TestNormalizeResponsesInputInstructions_PlainStringInput_NoOp(t *testing.T) {
	request := &dto.OpenAIResponsesRequest{
		Input: json.RawMessage(`"hi"`),
	}

	err := normalizeResponsesInputInstructions(request)
	require.NoError(t, err)
	require.Nil(t, request.Instructions)
}

func TestNormalizeResponsesInputInstructions_SkipsResponsesLiteBundle(t *testing.T) {
	// Responses Lite requests deliberately keep base instructions as a
	// developer message in `input` alongside an `additional_tools` bundle;
	// that shape must be left untouched.
	original := json.RawMessage(`[
		{"type":"additional_tools","role":"developer","tools":[]},
		{"role":"developer","content":"lite base instructions"},
		{"role":"user","content":"hi"}
	]`)
	request := &dto.OpenAIResponsesRequest{
		Input: original,
	}

	err := normalizeResponsesInputInstructions(request)
	require.NoError(t, err)
	require.Nil(t, request.Instructions)
	require.JSONEq(t, string(original), string(request.Input))
}

func TestNormalizeResponsesInputInstructions_PreservesTypedNonMessageDirective(t *testing.T) {
	request := &dto.OpenAIResponsesRequest{
		Input: json.RawMessage(`[
			{"type":"custom_directive","role":"developer","payload":"x"},
			{"role":"system","content":"real instructions"},
			{"role":"user","content":"hi"}
		]`),
	}

	err := normalizeResponsesInputInstructions(request)
	require.NoError(t, err)

	var instructions string
	require.NoError(t, json.Unmarshal(request.Instructions, &instructions))
	require.Equal(t, "real instructions", instructions)

	var input []map[string]any
	require.NoError(t, json.Unmarshal(request.Input, &input))
	require.Len(t, input, 2)
	require.Equal(t, "custom_directive", input[0]["type"])
	require.Equal(t, "user", input[1]["role"])
}
