from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Add normalization if needed
])

dataset = ImageTextDataset(image_paths, serbian_captions, tokenizer, image_transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fashion_clip_model.parameters(), lr=1e-5)

for epoch in range(num_epochs):
    for images, text_inputs in dataloader:
        images = images.to(device)
        text_inputs = {k: v.squeeze(1).to(device) for k, v in text_inputs.items()}

        # Forward pass
        outputs = fashion_clip_model(images=images, **text_inputs)

        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        ground_truth = torch.arange(len(images)).to(device)

        # Compute loss
        loss_i = criterion(logits_per_image, ground_truth)
        loss_t = criterion(logits_per_text, ground_truth)
        loss = (loss_i + loss_t) / 2

        # Update model
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()