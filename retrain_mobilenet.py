"""
Standalone script to retrain ONLY the MobileNet model with the optimized architecture.
Uses the same optimized build + training logic from main.py.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from main import (
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    NUM_CLASSES,
    MODEL_DIR,
    build_mobilenet,
    create_generators,
    get_callbacks,
    save_metadata,
    evaluate_model,
    plot_history,
    compile_model,
)


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)

    train_gen, val_gen, test_gen = create_generators()
    class_names = save_metadata(train_gen.class_indices)

    # Class weights
    labels = train_gen.classes
    weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(labels), y=labels
    )
    class_weights = dict(enumerate(weights))

    print("\n" + "=" * 60)
    print("Retraining MobileNet V2 Only  [Optimized]")
    print("=" * 60)

    model = build_mobilenet((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)
    compile_model(model, lr=1e-3, label_smoothing=0.05)

    # Phase 1 – head only (backbone frozen)
    PHASE1_EPOCHS = 10
    print(f"\nPhase 1: Warming up head ({PHASE1_EPOCHS} epochs, backbone frozen)…")
    model.fit(
        train_gen,
        epochs=PHASE1_EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks("mobilenet", use_lr_scheduler=True,
                                total_epochs=PHASE1_EPOCHS),
        class_weight=class_weights,
        verbose=1,
    )

    # Phase 2 – selective fine-tuning (last 30 layers)
    print("\nPhase 2: Selectively unfreezing last 30 MobileNetV2 layers…")
    backbone = [l for l in model.layers if isinstance(l, tf.keras.Model)][0]
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    for layer in backbone.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks("mobilenet", use_lr_scheduler=True,
                                finetune=True, total_epochs=EPOCHS),
        class_weight=class_weights,
        verbose=1,
    )

    # Load best checkpoint and evaluate
    best_model = tf.keras.models.load_model(MODEL_DIR / "mobilenet.keras")
    plot_history("MobileNet_retrained", history)
    evaluate_model("MobileNet (Optimized Retrain)", best_model, test_gen, class_names)

    print("\nDone. Updated model saved to:", MODEL_DIR / "mobilenet.keras")


if __name__ == "__main__":
    main()
