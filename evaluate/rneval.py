from evaluation.evaluate_classifier import ClassifierEvaluator

evaluator = ClassifierEvaluator(
    encoder=shared_encoder,
    classifier=class_predictor,
    device=device,
    num_classes=100
)

results = evaluator.evaluate(test_loader)

evaluator.print_results(results)

evaluator.save_metrics(
    results,
    save_dir="results"
)

evaluator.save_predictions(
    results,
    save_dir="results"
)

evaluator.save_features(
    results,
    save_dir="results"
)

evaluator.save_probabilities(
    results,
    save_dir="results"
)
