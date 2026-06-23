import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(ROOT_DIR)

from models.teacher_network import TeacherNetwork
from models.shared_encoder import SharedEncoder
from models.class_predictor import ClassPredictor
from models.domain_discriminator import DomainDiscriminator
from models.grl import RandomizedGRL

from losses.llcl_loss import LLCLLoss
from losses.feature_loss import FeatureLoss

# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(device)

# ==================================================
# LOAD DOMAIN DATASET
# ==================================================

domain_dataset = torch.load(
    os.path.join(
        ROOT_DIR,
        "data",
        "domain_dataset.pt"
    )
)

loader = DataLoader(
    domain_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2
)

# ==================================================
# LOAD TEACHER
# ==================================================

teacher_backbone = torch.hub.load(
    "pytorch/vision:v0.15.2",
    "resnet18",
    pretrained=False
)

teacher_backbone.fc = nn.Linear(
    teacher_backbone.fc.in_features,
    100
)

teacher_backbone.load_state_dict(
    torch.load(
        os.path.join(
            ROOT_DIR,
            "checkpoints",
            "teacher.pth"
        ),
        map_location=device
    )
)

teacher = TeacherNetwork(
    teacher_backbone
).to(device)

teacher.eval()

for p in teacher.parameters():

    p.requires_grad = False

# ==================================================
# SHARED ENCODER
# ==================================================

shared_encoder = SharedEncoder(
    teacher.encoder
).to(device)

# ==================================================
# CLASS PREDICTOR
# ==================================================

class_predictor = ClassPredictor(
    feature_dim=512,
    num_classes=100
).to(device)

# ==================================================
# DOMAIN DISCRIMINATOR
# ==================================================

domain_discriminator = DomainDiscriminator(
    feature_dim=512,
    num_domains=5
).to(device)

# ==================================================
# RANDOMIZED GRL
# ==================================================

grl = RandomizedGRL()

# ==================================================
# LOSSES
# ==================================================

classification_loss_fn = nn.CrossEntropyLoss()

domain_loss_fn = nn.CrossEntropyLoss()

feature_loss_fn = FeatureLoss()

llcl_loss_fn = LLCLLoss()

# ==================================================
# CLASS WEIGHTS
# ==================================================

class_counts = torch.zeros(100)

for _, labels, _ in loader:

    for y in labels:

        class_counts[y] += 1

class_weights = (

    class_counts.sum()
    /
    class_counts

)

class_weights = (
    class_weights
    /
    class_weights.mean()
)

class_weights = class_weights.to(device)

# ==================================================
# OPTIMIZER
# ==================================================

optimizer = optim.Adam(

    list(shared_encoder.parameters())

    +

    list(class_predictor.parameters())

    +

    list(domain_discriminator.parameters()),

    lr=1e-4,
    weight_decay=1e-5
)

# ==================================================
# HYPERPARAMETERS
# ==================================================

lambda_domain = 0.5

lambda_feat = 1.0

lambda_llcl = 0.5

epochs = 50

best_loss = 999999

# ==================================================
# TRAIN LOOP
# ==================================================

for epoch in range(epochs):

    shared_encoder.train()

    class_predictor.train()

    domain_discriminator.train()

    running_loss = 0

    pbar = tqdm(loader)

    for images, labels, domains in pbar:

        images = images.to(device)

        labels = labels.to(device)

        domains = domains.to(device)

        optimizer.zero_grad()

        # ==========================================
        # TEACHER BRANCH
        # ==========================================

        with torch.no_grad():

            teacher_features,\
            teacher_logits = teacher(
                images
            )

        # ==========================================
        # STUDENT BRANCH
        # ==========================================

        student_features = shared_encoder(
            images
        )

        student_logits = class_predictor(
            student_features
        )

        # ==========================================
        # CLASSIFICATION LOSS
        # ==========================================

        cls_loss = classification_loss_fn(

            student_logits,

            labels
        )

        # ==========================================
        # FEATURE CONSISTENCY
        # ==========================================

        feat_loss = feature_loss_fn(

            teacher_features,

            student_features
        )

        # ==========================================
        # LLCL LOSS
        # ==========================================

        llcl_loss = llcl_loss_fn(

            teacher_logits,

            student_logits,

            class_weights
        )

        # ==========================================
        # RANDOMIZED GRL
        # ==========================================

        reversed_features = grl(

            student_features,

            epoch,

            epochs
        )

        # ==========================================
        # DOMAIN PREDICTION
        # ==========================================

        domain_logits = (

            domain_discriminator(

                reversed_features
            )
        )

        domain_loss = domain_loss_fn(

            domain_logits,

            domains
        )

        # ==========================================
        # TOTAL LOSS
        # ==========================================

        total_loss = (

              cls_loss

            + lambda_domain
              * domain_loss

            + lambda_feat
              * feat_loss

            + lambda_llcl
              * llcl_loss
        )

        total_loss.backward()

        optimizer.step()

        running_loss += total_loss.item()

        pbar.set_postfix(

            CLS=f"{cls_loss.item():.4f}",

            DOM=f"{domain_loss.item():.4f}",

            FEAT=f"{feat_loss.item():.4f}",

            LLCL=f"{llcl_loss.item():.4f}",

            TOTAL=f"{total_loss.item():.4f}"
        )

    epoch_loss = (

        running_loss
        /
        len(loader)
    )

    print(
        f"\nEpoch {epoch+1}"
    )

    print(
        f"Loss = {epoch_loss:.4f}"
    )

    # ==============================================
    # SAVE BEST MODEL
    # ==============================================

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(

            {

                "shared_encoder":

                shared_encoder.state_dict(),

                "class_predictor":

                class_predictor.state_dict(),

                "domain_discriminator":

                domain_discriminator.state_dict()
            },

            os.path.join(

                ROOT_DIR,

                "checkpoints",

                "domain_adaptation_best.pth"
            )
        )

        print(
            "Best model saved."
        )

print(
    "\nTraining Finished"
)