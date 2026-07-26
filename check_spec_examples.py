from pinoybot import tag_language


examples = [
    (
        [
            "23", "na", "flood", "control", "projects",
            "ang", "na-award", "sa", "St", ".", "Gerrard",
            "Construction", "."
        ],
        [
            "OTH", "FIL", "ENG", "ENG", "ENG",
            "FIL", "CS", "FIL", "OTH", "OTH", "OTH",
            "ENG", "OTH"
        ]
    ),
    (
        [
            "Ang", "Chinito", "Walkers", "ay", "mga",
            "sikat", "na", "content", "creators", "mula",
            "sa", "DLSU", "."
        ],
        [
            "FIL", "FIL", "ENG", "FIL", "FIL",
            "FIL", "FIL", "ENG", "ENG", "FIL",
            "FIL", "OTH", "OTH"
        ]
    ),
    (
        [
            "Madami", "ang", "nag-march", "sa", "13",
            "trillion", "peso", "march", "sa", "EDSA",
            "monument", "at", "Luneta", "Park", "."
        ],
        [
            "FIL", "FIL", "CS", "FIL", "OTH",
            "ENG", "ENG", "ENG", "FIL", "OTH",
            "ENG", "FIL", "OTH", "ENG", "OTH"
        ]
    )
]


total_correct = 0
total_tokens = 0

for example_number, (tokens, expected) in enumerate(examples, start=1):
    predicted = tag_language(tokens)

    print(f"\nEXAMPLE {example_number}")
    print(f"{'Token':<15} {'Expected':<10} {'Predicted':<10} Result")
    print("-" * 50)

    for token, expected_tag, predicted_tag in zip(
        tokens,
        expected,
        predicted
    ):
        is_correct = expected_tag == predicted_tag
        result = "OK" if is_correct else "WRONG"

        if is_correct:
            total_correct += 1

        total_tokens += 1

        print(
            f"{token:<15} "
            f"{expected_tag:<10} "
            f"{predicted_tag:<10} "
            f"{result}"
        )

print("\n" + "=" * 50)
print(f"Correct: {total_correct}/{total_tokens}")
print(f"Accuracy: {total_correct / total_tokens:.2%}")