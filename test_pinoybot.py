import unittest

from pinoybot import tag_language


class TestPinoyBot(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(tag_language([]), [])

    def test_output_length(self):
        tokens = ["Love", "kita", "."]
        tags = tag_language(tokens)

        self.assertEqual(len(tags), len(tokens))

    def test_allowed_tags(self):
        tokens = ["Love", "kita", ".", "nag-march", "DLSU"]
        tags = tag_language(tokens)

        allowed_tags = {"ENG", "FIL", "CS", "OTH"}

        for tag in tags:
            self.assertIn(tag, allowed_tags)

    def test_specification_example(self):
        tokens = ["Love", "kita", "."]
        expected = ["ENG", "FIL", "OTH"]

        self.assertEqual(tag_language(tokens), expected)

    def test_code_switched_word(self):
        tokens = ["nag-march"]

        self.assertEqual(tag_language(tokens), ["CS"])

    def test_abbreviation(self):
        tokens = ["DLSU"]

        self.assertEqual(tag_language(tokens), ["OTH"])

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            tag_language("Love kita")

    def test_non_string_token(self):
        with self.assertRaises(TypeError):
            tag_language(["Love", 123])


if __name__ == "__main__":
    unittest.main()