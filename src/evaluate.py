def evaluate_model(model, test_ds):
    loss, acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {acc:.4f}")
    return acc