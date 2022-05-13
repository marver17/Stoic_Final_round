import shutil
from process import do_process
from algorithm.lesion_segmentation import blockPrint
from train import do_train
DATA_DIR = "/input/"
ARTIFACT_DIR = "/output/"


if __name__ == "__main__":
    # Substitute do_learning for your training function.
    # It is recommended to write artifacts (e.g. model weights) to ARTIFACT_DIR during training.
    # artifacts = do_learning(DATA_DIR, ARTIFACT_DIR)
    blockPrint()
    do_process(DATA_DIR)
    a,b = do_train()

    # When the learning has completed, any artifacts should have been saved to ARTIFACT_DIR.
    # Alternatively, you can copy artifacts to ARTIFACT_DIR after the learning has completed:
    # for artifact in artifacts:
    #     shutil.copy(artifact, ARTIFACT_DIR)
     
    # print("Training completed.")
