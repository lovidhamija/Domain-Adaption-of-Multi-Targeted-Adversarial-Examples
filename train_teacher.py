

from torch.utils.data import DataLoader


# =====================================================
# PATH SETUP
# =====================================================

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

CHECKPOINT_DIR = os.path.join(
    ROOT_DIR,
    "checkpoints"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# =====================================================
# DEVICE
# =====================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"
)

print(f"Using Device: {device}")


# =====================================================
# HYPERPARAMETERS
# =====================================================

BATCH_SIZE = 128

NUM_EPOCHS = 50

LEARNING_RATE = 1e-3

NUM_CLASSES = 100


# =====================================================
# TRANSFORMS
# =====================================================

train_transform = transforms.Compose([

    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=(0.5071, 0.4867, 0.4408),

        std=(0.2675, 0.2565, 0.2761)
    )
])

test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(

        mean=(0.5071, 0.4867, 0.4408),

        std=(0.2675, 0.2565, 0.2761)
    )
])


# =====================================================
# DATASET
# =====================================================

train_dataset = datasets.CIFAR100(

    root=os.path.join(
        ROOT_DIR,
        "data"
    ),

    train=True,

    download=True,

    transform=train_transform
)

test_dataset = datasets.CIFAR100(

    root=os.path.join(
        ROOT_DIR,
        "data"
    ),

    train=False,

    download=True,

    transform=test_transform
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=2,

    pin_memory=False
)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=2,

    pin_memory=True
)


# =====================================================
# MODEL
# =====================================================

model = resnet18(weights=None)

model.fc = nn.Linear(

    model.fc.in_features,

    NUM_CLASSES
)

model = model.to(device)


# =====================================================
# LOSS
# =====================================================

criterion = nn.CrossEntropyLoss()


# =====================================================
# OPTIMIZER
# =====================================================

optimizer = optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4
)


# =====================================================
# LR SCHEDULER
# =====================================================

scheduler = optim.lr_scheduler.StepLR(

    optimizer,

    step_size=20,

    gamma=0.1
)


# =====================================================
# TRAIN FUNCTION
# =====================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    pbar = tqdm(

        train_loader,

        desc="Training"
    )

    for images, labels in pbar:

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(

            outputs,

            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)

        correct += (

            preds == labels

        ).sum().item()

        total += labels.size(0)

        acc = (

            100.0 * correct

            / total
        )

        pbar.set_postfix(

            Loss=f"{loss.item():.4f}",

            Acc=f"{acc:.2f}"
        )

    epoch_loss = (

        running_loss

        / len(train_loader)
    )

    epoch_acc = (

        100.0 * correct

        / total
    )

    return epoch_loss, epoch_acc


# =====================================================
# VALIDATION FUNCTION
# =====================================================

@torch.no_grad()

def validate():

    model.eval()

    correct = 0

    total = 0

    running_loss = 0

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(

            outputs,

            labels
        )

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)

        correct += (

            preds == labels

        ).sum().item()

        total += labels.size(0)

    val_loss = (

        running_loss

        / len(test_loader)
    )

    val_acc = (

        100.0 * correct

        / total
    )

    return val_loss, val_acc


# =====================================================
# TRAINING LOOP
# =====================================================

best_acc = 0.0

for epoch in range(NUM_EPOCHS):

    print(
        f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]"
    )

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc = validate()

    scheduler.step()

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Acc : {train_acc:.2f}%"
    )

    print(
        f"Val Loss  : {val_loss:.4f}"
    )

    print(
        f"Val Acc   : {val_acc:.2f}%"
    )

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(

            model.state_dict(),

            os.path.join(

                CHECKPOINT_DIR,

                "teacher.pth"
            )
        )

        print(
            "Best Teacher Saved"
        )


print(
    f"\nBest Accuracy: {best_acc:.2f}%"
)