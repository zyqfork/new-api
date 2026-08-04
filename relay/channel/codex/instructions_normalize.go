package codex

import (
	"encoding/json"
	"strings"

	"github.com/QuantumNous/new-api/relaykit/dto"
)

// normalizeResponsesInputInstructions hoists leading system/developer "message"
// input items into the top-level Instructions field.
//
// Codex-native clients (Codex CLI, and Pi's own openai-codex provider) always
// send the system prompt as a distinct top-level `instructions` string and
// keep `input` free of system/developer messages. Generic OpenAI Responses
// clients (e.g. Pi's openai-responses provider, used against non-Codex
// providers) instead embed the system prompt as the first `input` message
// with role "system" or "developer" and never populate `instructions`.
//
// Forwarding that generic shape to the Codex backend unmodified produces a
// structurally different request than what Codex-native traffic sends: the
// system preamble ends up in `input` instead of `instructions`. Codex's own
// request handling - and by extension its prompt-cache prefix matching -
// treats `instructions` as a distinct, stable segment from `input`, so this
// shape mismatch can reduce prompt-cache hit reliability even though the
// content is semantically identical. This mirrors the "instruction hoisting"
// step used by other Responses-API-compatible Codex proxies (e.g.
// github.com/Soju06/codex-lb) to keep the wire shape Codex-native.
//
// Responses Lite requests deliberately carry their tool bundle as an `input`
// item with type "additional_tools" and keep base instructions as a developer
// message in `input` with an empty top-level `instructions`; that shape is
// exactly what the Lite upstream expects, so hoisting is skipped whenever an
// `additional_tools` item is present.
func normalizeResponsesInputInstructions(request *dto.OpenAIResponsesRequest) error {
	if len(request.Input) == 0 {
		return nil
	}

	var rawItems []json.RawMessage
	if err := json.Unmarshal(request.Input, &rawItems); err != nil {
		// Not a JSON array (e.g. a plain string `input`) - nothing to hoist.
		return nil
	}

	items := make([]map[string]any, 0, len(rawItems))
	for _, raw := range rawItems {
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			// Non-object array item - bail out rather than risk corrupting
			// a shape we don't understand.
			return nil
		}
		items = append(items, m)
	}

	if responsesInputUsesLiteTools(items) {
		return nil
	}

	var instructionParts []string
	nextItems := make([]map[string]any, 0, len(items))
	changed := false

	for _, item := range items {
		if isPreservedNonMessageDirective(item) {
			// Non-message typed system/developer items (e.g. the Codex
			// responses-lite additional_tools bundle) carry no instruction
			// content; pass them through untouched.
			nextItems = append(nextItems, item)
			changed = true

			continue
		}

		role, _ := item["role"].(string)
		if role != "system" && role != "developer" {
			nextItems = append(nextItems, item)
			continue
		}

		instructionText, preservedContent := splitResponsesInstructionItemContent(item)
		if instructionText != "" {
			instructionParts = append(instructionParts, instructionText)
		}

		if preservedContent != nil {
			preserved := make(map[string]any, len(item))
			for k, v := range item {
				preserved[k] = v
			}

			preserved["role"] = "user"
			preserved["content"] = preservedContent
			nextItems = append(nextItems, preserved)
		}

		changed = true
	}

	if !changed {
		return nil
	}

	var existingInstructions string
	if len(request.Instructions) > 0 {
		_ = json.Unmarshal(request.Instructions, &existingInstructions)
	}

	merged := mergeResponsesInstructions(existingInstructions, instructionParts)

	mergedBytes, err := json.Marshal(merged)
	if err != nil {
		return err
	}

	itemsBytes, err := json.Marshal(nextItems)
	if err != nil {
		return err
	}

	request.Instructions = mergedBytes
	request.Input = itemsBytes

	return nil
}

func responsesInputUsesLiteTools(items []map[string]any) bool {
	for _, item := range items {
		if t, _ := item["type"].(string); t == "additional_tools" {
			return true
		}
	}

	return false
}

// isPreservedNonMessageDirective reports whether item is a system/developer
// input item whose explicit type is something other than "message" (e.g. the
// Codex responses-lite additional_tools bundle). Such items carry no
// instruction text and must be forwarded byte-identical.
func isPreservedNonMessageDirective(item map[string]any) bool {
	role, _ := item["role"].(string)
	if role != "system" && role != "developer" {
		return false
	}

	itemType, hasType := item["type"]
	if !hasType || itemType == nil {
		return false
	}

	typeStr, _ := itemType.(string)

	return typeStr != "" && typeStr != "message"
}

// splitResponsesInstructionItemContent extracts hoistable instruction text
// from a system/developer message item's content, and returns any remaining
// content (e.g. an embedded image part) that cannot be represented as plain
// instruction text and must be preserved as a regular input message instead
// of being silently dropped.
func splitResponsesInstructionItemContent(item map[string]any) (string, any) {
	content, hasContent := item["content"]
	if !hasContent || content == nil {
		return "", nil
	}

	if text, ok := content.(string); ok {
		return text, nil
	}

	if parts, ok := content.([]any); ok {
		var instructionParts []string

		var preservedParts []any

		for _, part := range parts {
			if text, ok := responsesInstructionContentText(part); ok {
				if text != "" {
					instructionParts = append(instructionParts, text)
				}

				continue
			}

			preservedParts = append(preservedParts, part)
		}

		var preserved any
		if len(preservedParts) > 0 {
			preserved = preservedParts
		}

		return strings.Join(instructionParts, "\n"), preserved
	}

	if text, ok := responsesInstructionContentText(content); ok {
		return text, nil
	}

	return "", content
}

// responsesInstructionContentText reports whether content is directly
// text-extractable (a plain string, or an object carrying a string "text"
// field, matching input_text/text/output_text-shaped content parts) and
// returns that text. The second return value is false when content is some
// other shape (e.g. an image part) that must be preserved instead.
func responsesInstructionContentText(content any) (string, bool) {
	if text, ok := content.(string); ok {
		return text, true
	}

	m, ok := content.(map[string]any)
	if !ok {
		return "", false
	}

	text, ok := m["text"].(string)
	if !ok {
		return "", false
	}

	return text, true
}

func mergeResponsesInstructions(existing string, extraParts []string) string {
	nonEmpty := make([]string, 0, len(extraParts))

	for _, p := range extraParts {
		if p != "" {
			nonEmpty = append(nonEmpty, p)
		}
	}

	extra := strings.Join(nonEmpty, "\n")
	if extra == "" {
		return existing
	}

	if existing == "" {
		return extra
	}

	return existing + "\n" + extra
}
