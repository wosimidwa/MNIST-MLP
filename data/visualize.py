import matplotlib.pyplot as plt

def show_images(images, titles):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(15,10))
    for index, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, index+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
