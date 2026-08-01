import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.apertus2608_detector import Apertus2608Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

WEATHER_CALL = '<|tools_prefix|>[{"get_weather": {"city": "Paris"}}]'
SUFFIX = "<|tools_suffix|>"


class TestApertus2608Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="lookup_user",
                    description="Look up a user by name",
                    parameters={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                        },
                        "required": ["name"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="ping",
                    description="Ping the server. Takes no arguments.",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]
        self.detector = Apertus2608Detector()

    def _calls(self, result):
        return [(c.name, json.loads(c.parameters)) for c in result.calls]

    def _stream(self, text, chunk_size):
        """Feed text in fixed-size chunks; return (normal_text, [(name, args)])."""
        detector = Apertus2608Detector()
        normal = ""
        calls = {}  # tool_index -> {"name": ..., "args": accumulated str}
        for i in range(0, len(text), chunk_size):
            result = detector.parse_streaming_increment(
                text[i : i + chunk_size], self.tools
            )
            normal += result.normal_text
            for c in result.calls:
                entry = calls.setdefault(c.tool_index, {"name": None, "args": ""})
                if c.name is not None:
                    entry["name"] = c.name
                entry["args"] += c.parameters or ""
        return normal, [
            (calls[i]["name"], json.loads(calls[i]["args"])) for i in sorted(calls)
        ]

    def test_has_tool_call_true(self):
        self.assertTrue(self.detector.has_tool_call(WEATHER_CALL))

    def test_has_tool_call_false(self):
        self.assertFalse(self.detector.has_tool_call("The weather in Paris is sunny."))

    def test_trimmed_suffix_single_call(self):
        """The real-server case: suffix consumed as the EOS stop and trimmed."""
        result = self.detector.detect_and_parse(WEATHER_CALL, self.tools)
        self.assertEqual(self._calls(result), [("get_weather", {"city": "Paris"})])
        self.assertEqual(result.normal_text, "")

    def test_explicit_suffix_still_parses(self):
        """Servers that keep stop tokens (no_stop_trim) deliver the suffix."""
        result = self.detector.detect_and_parse(WEATHER_CALL + SUFFIX, self.tools)
        self.assertEqual(self._calls(result), [("get_weather", {"city": "Paris"})])
        self.assertEqual(result.normal_text, "")

    def test_trimmed_suffix_parallel_calls(self):
        """One call list with several entries = OpenAI parallel tool calls."""
        text = (
            '<|tools_prefix|>[{"get_weather": {"city": "Bern"}}, '
            '{"lookup_user": {"name": "amy"}}]'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(
            self._calls(result),
            [("get_weather", {"city": "Bern"}), ("lookup_user", {"name": "amy"})],
        )

    def test_trimmed_suffix_empty_args(self):
        result = self.detector.detect_and_parse(
            '<|tools_prefix|>[{"ping": {}}]', self.tools
        )
        self.assertEqual(self._calls(result), [("ping", {})])

    def test_null_args_becomes_empty_object(self):
        """A no-argument call may be emitted as null; normalize to {}."""
        result = self.detector.detect_and_parse(
            '<|tools_prefix|>[{"ping": null}]', self.tools
        )
        self.assertEqual(self._calls(result), [("ping", {})])

    def test_two_blocks_with_suffixes(self):
        """Two suffix-terminated blocks (no_stop_trim serving) both parse;
        the text between them is kept."""
        text = (
            WEATHER_CALL + SUFFIX + " and " + '<|tools_prefix|>[{"ping": {}}]' + SUFFIX
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(
            self._calls(result), [("get_weather", {"city": "Paris"}), ("ping", {})]
        )
        self.assertEqual(result.normal_text, "and")

    def test_text_before_call_trimmed_suffix(self):
        result = self.detector.detect_and_parse(
            "Let me check. " + WEATHER_CALL, self.tools
        )
        self.assertEqual(self._calls(result), [("get_weather", {"city": "Paris"})])
        self.assertEqual(result.normal_text, "Let me check.")

    def test_trailing_whitespace_after_call(self):
        result = self.detector.detect_and_parse(WEATHER_CALL + "\n", self.tools)
        self.assertEqual(self._calls(result), [("get_weather", {"city": "Paris"})])

    def test_plain_text_untouched(self):
        text = "It is sunny in Paris."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, text)

    def test_truncated_json_stays_normal_text(self):
        """Cut off mid-arguments (max_tokens): not a parsable call."""
        text = '<|tools_prefix|>[{"get_weather": {"ci'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, text)

    def test_trailing_prose_after_array_stays_normal_text(self):
        """Complete array followed by non-suffix prose and NO suffix anywhere:
        the implicit-suffix rule must not fire (only end-of-text counts as a
        trimmed stop), so the whole thing stays normal text."""
        text = WEATHER_CALL + " and then some prose"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, text)

    def test_prose_between_array_and_suffix_skipped(self):
        """With an explicit suffix present, junk between the array and the
        suffix is skipped and the call is still emitted."""
        text = WEATHER_CALL + " oops" + SUFFIX
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(self._calls(result), [("get_weather", {"city": "Paris"})])
        self.assertEqual(result.normal_text, "")

    def test_unknown_tool_dropped(self):
        """An undefined tool name is dropped entirely (call filtered, text
        consumed) — same silent-drop semantics as 2509 with a suffix."""
        result = self.detector.detect_and_parse(
            '<|tools_prefix|>[{"rm_rf": {"path": "/"}}]', self.tools
        )
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "")

    def test_structure_info_forced_format(self):
        """The grammar path (tool_choice required/named) builds constrained
        output from structure_info(); its begin/args/end concatenation must
        round-trip through the detector."""
        info = self.detector.structure_info()("get_weather")
        self.assertEqual(info.trigger, "<|tools_prefix|>")
        forced = info.begin + '{"city": "Paris"}' + info.end
        result = Apertus2608Detector().detect_and_parse(forced, self.tools)
        self.assertEqual(self._calls(result), [("get_weather", {"city": "Paris"})])

    def test_stream_matches_non_stream(self):
        """Property: for well-formed output, streaming accumulation and
        detect_and_parse agree on calls and (stripped) normal text."""
        corpus = [
            WEATHER_CALL,
            WEATHER_CALL + SUFFIX,
            "Checking now. " + WEATHER_CALL,
            WEATHER_CALL + "\n",
            '<|tools_prefix|>[{"get_weather": {"city": "Bern"}}, {"ping": {}}]',
            WEATHER_CALL + SUFFIX + " and " + '<|tools_prefix|>[{"ping": {}}]' + SUFFIX,
            "It is sunny in Paris.",
        ]
        for text in corpus:
            for chunk_size in (1, 7):
                with self.subTest(text=text[:40], chunk_size=chunk_size):
                    non_stream = Apertus2608Detector().detect_and_parse(
                        text, self.tools
                    )
                    normal, calls = self._stream(text, chunk_size)
                    self.assertEqual(calls, self._calls(non_stream))
                    self.assertEqual(normal.strip(), non_stream.normal_text)

    def test_stream_trimmed_suffix(self):
        for chunk_size in (1, 3, 7, 1000):
            with self.subTest(chunk_size=chunk_size):
                normal, calls = self._stream(WEATHER_CALL, chunk_size)
                self.assertEqual(calls, [("get_weather", {"city": "Paris"})])
                self.assertEqual(normal, "")

    def test_stream_explicit_suffix_not_leaked(self):
        for chunk_size in (1, 3, 7, 1000):
            with self.subTest(chunk_size=chunk_size):
                normal, calls = self._stream(WEATHER_CALL + SUFFIX, chunk_size)
                self.assertEqual(calls, [("get_weather", {"city": "Paris"})])
                self.assertEqual(normal, "")

    def test_stream_text_then_call(self):
        for chunk_size in (1, 3, 7, 1000):
            with self.subTest(chunk_size=chunk_size):
                normal, calls = self._stream(
                    "Checking now. " + WEATHER_CALL, chunk_size
                )
                self.assertEqual(calls, [("get_weather", {"city": "Paris"})])
                self.assertEqual(normal, "Checking now. ")

    def test_stream_parallel_calls(self):
        """One call list with several entries = OpenAI parallel tool calls."""
        text = '<|tools_prefix|>[{"get_weather": {"city": "Bern"}}, {"ping": {}}]'
        for chunk_size in (1, 5, 1000):
            with self.subTest(chunk_size=chunk_size):
                normal, calls = self._stream(text, chunk_size)
                self.assertEqual(
                    calls, [("get_weather", {"city": "Bern"}), ("ping", {})]
                )

    def test_stream_two_blocks_with_suffixes(self):
        """Two suffix-terminated blocks parse across token-sized increments.
        (A whole trailing block inside one final coarse increment would be
        held for a next increment that never comes — a limitation inherited
        from the 2509 streaming design, so only small chunks are asserted.)"""
        text = (
            WEATHER_CALL + SUFFIX + " and " + '<|tools_prefix|>[{"ping": {}}]' + SUFFIX
        )
        for chunk_size in (1, 3):
            with self.subTest(chunk_size=chunk_size):
                normal, calls = self._stream(text, chunk_size)
                self.assertEqual(
                    calls, [("get_weather", {"city": "Paris"}), ("ping", {})]
                )
                self.assertEqual(normal, " and ")

    def test_stream_plain_text(self):
        text = "It is sunny in Paris."
        for chunk_size in (1, 4, 1000):
            with self.subTest(chunk_size=chunk_size):
                normal, calls = self._stream(text, chunk_size)
                self.assertEqual(calls, [])
                self.assertEqual(normal, text)

    def test_stream_emission_order_name_then_args(self):
        """OpenAI streaming semantics: for each call, the first emitted item
        carries the tool name with empty parameters, the next carries the full
        JSON arguments with no name, and both share one tool_index."""
        detector = Apertus2608Detector()
        detector.parse_streaming_increment(WEATHER_CALL[:-1], self.tools)
        result = detector.parse_streaming_increment(WEATHER_CALL[-1], self.tools)
        self.assertEqual(len(result.calls), 2)
        first, second = result.calls
        self.assertEqual(first.name, "get_weather")
        self.assertEqual(first.parameters, "")
        self.assertIsNone(second.name)
        self.assertEqual(json.loads(second.parameters), {"city": "Paris"})
        self.assertEqual(first.tool_index, second.tool_index)

    def test_stream_call_emitted_when_array_closes(self):
        """The fix's streaming core: the call must be emitted in the SAME
        increment that completes the JSON list — the trimmed <|tools_suffix|>
        never arrives, so waiting for it would swallow the call."""
        detector = Apertus2608Detector()
        before = detector.parse_streaming_increment(WEATHER_CALL[:-1], self.tools)
        self.assertEqual(before.calls, [])
        at_close = detector.parse_streaming_increment(WEATHER_CALL[-1], self.tools)
        self.assertTrue(at_close.calls)

    def test_stream_parallel_call_indexes(self):
        """Parallel calls in one list get distinct, ordered tool indexes."""
        text = '<|tools_prefix|>[{"get_weather": {"city": "Bern"}}, {"ping": {}}]'
        detector = Apertus2608Detector()
        calls = []
        for i in range(0, len(text), 5):
            calls += detector.parse_streaming_increment(
                text[i : i + 5], self.tools
            ).calls
        named = [(c.tool_index, c.name) for c in calls if c.name]
        self.assertEqual(named, [(0, "get_weather"), (1, "ping")])

    def test_stream_whitespace_before_suffix_swallowed(self):
        """Whitespace between the list and the suffix is held, then dropped
        with the suffix — never streamed into content."""
        text = WEATHER_CALL + "\n" + SUFFIX
        for chunk_size in (1, 4, 1000):
            with self.subTest(chunk_size=chunk_size):
                normal, calls = self._stream(text, chunk_size)
                self.assertEqual(calls, [("get_weather", {"city": "Paris"})])
                self.assertEqual(normal, "")

    def test_stream_prefix_lookalike_prose_released(self):
        """Prose containing a partial-prefix lookalike (`<|tools`) is held
        while ambiguous, then released verbatim once it diverges."""
        text = "see the <|tools token here"
        for chunk_size in (1, 3, 1000):
            with self.subTest(chunk_size=chunk_size):
                normal, calls = self._stream(text, chunk_size)
                self.assertEqual(calls, [])
                self.assertEqual(normal, text)

    def test_stream_suffix_in_later_increment(self):
        """Suffix arrives whole increments after the array closed."""
        detector = Apertus2608Detector()
        r1 = detector.parse_streaming_increment(WEATHER_CALL, self.tools)
        self.assertEqual([c.name for c in r1.calls if c.name], ["get_weather"])
        r2 = detector.parse_streaming_increment(SUFFIX[:7], self.tools)
        r3 = detector.parse_streaming_increment(SUFFIX[7:], self.tools)
        self.assertEqual(r2.normal_text + r3.normal_text, "")
        self.assertEqual(r2.calls, [])
        self.assertEqual(r3.calls, [])

    def test_stream_text_after_swallowed_suffix(self):
        """Prose following the suffix must still stream out."""
        detector = Apertus2608Detector()
        detector.parse_streaming_increment(WEATHER_CALL, self.tools)
        result = detector.parse_streaming_increment(SUFFIX + "Done.", self.tools)
        self.assertEqual(result.normal_text, "Done.")


if __name__ == "__main__":
    import unittest

    unittest.main()
