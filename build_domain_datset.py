import os
import sys
import torch
import torch.nn as nn

from tqdm import tqdm

from torchvision import datasets
from torchvision import transforms
from torchvision.models import resnet18

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import torchattacks


# =====================================================
# PATHS
# =====================================================

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DATA_DIR = os.path.join(
    ROOT_DIR,
    "data"
)

CHECKPOINT_DIR = os.path.join(
    ROOT_DIR,
    "checkpoints"
)

os.makedirs(DATA_DIR, exist_ok=True)


# =====================================================
# DEVICE
# =====================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"
)

print(
    f"Using Device: {device}"
)


# =====================================================
# CIFAR100
# =====================================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(

        mean=(0.5071, 0.4867, 0.4408),

        std=(0.2675, 0.2565, 0.2761)
    )
])


dataset = datasets.CIFAR100(

    root=DATA_DIR,

    train=True,

    download=True,

    transform=transform
)

loader = DataLoader(

    dataset,

    batch_size=128,

    shuffle=False,

    num_workers=2
)


# =====================================================
# LOAD TEACHER MODEL
# =====================================================

model = resnet18(weights=None)

model.fc = nn.Linear(

    model.fc.in_features,

    100
)

model.load_state_dict(

    torch.load(

        os.path.join(

            CHECKPOINT_DIR,

            "teacher.pth"
        ),

        map_location=device
    )
)

model = model.to(device)

model.eval()


# =====================================================
# ATTACKS
# =====================================================

fgsm_attack = torchattacks.FGSM(

    model,

    eps=8/255
)

pgd_attack = torchattacks.PGD(

    model,

    eps=8/255,

    alpha=2/255,

    steps=10
)

cw_attack = torchattacks.CW(

    model,

    c=1,

    kappa=0,

    steps=50
)


# =====================================================
# PATCH ATTACK
# =====================================================

def apply_patch(

    images,

    patch_size=8
):

    patched = images.clone()

    patched[
        :,
        :,
        0:patch_size,
        0:patch_size
    ] = 1.0

    return patched


# =====================================================
# STORAGE
# =====================================================

clean_images = []
clean_labels = []

fgsm_images = []
fgsm_labels = []

pgd_images = []
pgd_labels = []

cw_images = []
cw_labels = []

patch_images = []
patch_labels = []


# =====================================================
# GENERATE DATASETS
# =====================================================

for images, labels in tqdm(loader):

    images = images.to(device)

    labels = labels.to(device)

    ##################################################
    # CLEAN
    ##################################################

    clean_images.append(

        images.cpu()
    )

    clean_labels.append(

        labels.cpu()
    )

    ##################################################
    # FGSM
    ##################################################

    adv_fgsm = fgsm_attack(

        images,

        labels
    )

    fgsm_images.append(

        adv_fgsm.cpu()
    )

    fgsm_labels.append(

        labels.cpu()
    )

    ##################################################
    # PGD
    ##################################################

    adv_pgd = pgd_attack(

        images,

        labels
    )

    pgd_images.append(

        adv_pgd.cpu()
    )

    pgd_labels.append(

        labels.cpu()
    )

    ##################################################
    # CW
    ##################################################

    adv_cw = cw_attack(

        images,

        labels
    )

    cw_images.append(

        adv_cw.cpu()
    )

    cw_labels.append(

        labels.cpu()
    )

    ##################################################
    # PATCH
    ##################################################

    adv_patch = apply_patch(
        images
    )

    patch_images.append(

        adv_patch.cpu()
    )

    patch_labels.append(

        labels.cpu()
    )


# =====================================================
# CONCATENATE
# =====================================================

clean_images = torch.cat(clean_images)

clean_labels = torch.cat(clean_labels)

fgsm_images = torch.cat(fgsm_images)

fgsm_labels = torch.cat(fgsm_labels)

pgd_images = torch.cat(pgd_images)

pgd_labels = torch.cat(pgd_labels)

cw_images = torch.cat(cw_images)

cw_labels = torch.cat(cw_labels)

patch_images = torch.cat(patch_images)

patch_labels = torch.cat(patch_labels)


# =====================================================
# SAVE INDIVIDUAL DATASETS
# =====================================================

torch.save(

    clean_images,

    os.path.join(
        DATA_DIR,
        "clean_images.pt"
    )
)

torch.save(

    clean_labels,

    os.path.join(
        DATA_DIR,
        "clean_labels.pt"
    )
)

torch.save(

    fgsm_images,

    os.path.join(
        DATA_DIR,
        "fgsm_images.pt"
    )
)

torch.save(

    fgsm_labels,

    os.path.join(
        DATA_DIR,
        "fgsm_labels.pt"
    )
)

torch.save(

    pgd_images,

    os.path.join(
        DATA_DIR,
        "pgd_images.pt"
    )
)

torch.save(

    pgd_labels,

    os.path.join(
        DATA_DIR,
        "pgd_labels.pt"
    )
)

torch.save(

    cw_images,

    os.path.join(
        DATA_DIR,
        "cw_images.pt"
    )
)

torch.save(

    cw_labels,

    os.path.join(
        DATA_DIR,
        "cw_labels.pt"
    )
)

torch.save(

    patch_images,

    os.path.join(
        DATA_DIR,
        "patch_images.pt"
    )
)

torch.save(

    patch_labels,

    os.path.join(
        DATA_DIR,
        "patch_labels.pt"
    )
)

print(
    "All attack datasets saved."
)


# =====================================================
# DOMAIN LABELS
# =====================================================

clean_domains = torch.zeros(

    len(clean_labels),

    dtype=torch.long
)

fgsm_domains = torch.ones(

    len(fgsm_labels),

    dtype=torch.long
)

pgd_domains = torch.full(

    (len(pgd_labels),),

    2,

    dtype=torch.long
)

cw_domains = torch.full(

    (len(cw_labels),),

    3,

    dtype=torch.long
)

patch_domains = torch.full(

    (len(patch_labels),),

    4,

    dtype=torch.long
)


# =====================================================
# BUILD DOMAIN DATASET
# =====================================================

all_images = torch.cat([

    clean_images,

    fgsm_images,

    pgd_images,

    cw_images,

    patch_images
])

all_labels = torch.cat([

    clean_labels,

    fgsm_labels,

    pgd_labels,

    cw_labels,

    patch_labels
])

all_domains = torch.cat([

    clean_domains,

    fgsm_domains,

    pgd_domains,

    cw_domains,

    patch_domains
])


domain_dataset = TensorDataset(

    all_images,

    all_labels,

    all_domains
)


torch.save(

    domain_dataset,

    os.path.join(
        DATA_DIR,
        "domain_dataset.pt"
    )
)

print(
    "domain_dataset.pt saved successfully"
)