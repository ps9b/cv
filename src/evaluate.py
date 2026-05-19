import os


transform = transforms.ToTensor()


def evaluate_metrics(clean, output):
    clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    p = psnr(clean_gray, output_gray)
    s = ssim(clean_gray, output_gray)

    return p, s


# DnCNN 평가

def evaluate_dncnn(clean_dir, noisy_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DnCNN().to(device)
    model.load_state_dict(torch.load('../models/dncnn.pth'))

    model.eval()

    psnr_list = []
    ssim_list = []

    for filename in os.listdir(clean_dir):
        clean = cv2.imread(os.path.join(clean_dir, filename))
        noisy = cv2.imread(os.path.join(noisy_dir, filename))

        noisy_rgb = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)

        tensor = transform(noisy_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)

        output = output.squeeze(0).cpu().numpy()
        output = np.transpose(output, (1, 2, 0))
        output = np.clip(output * 255, 0, 255).astype(np.uint8)

        p, s = evaluate_metrics(clean, output)

        psnr_list.append(p)
        ssim_list.append(s)

    print('DnCNN Results')
    print('Average PSNR:', np.mean(psnr_list))
    print('Average SSIM:', np.mean(ssim_list))


if __name__ == '__main__':
    evaluate_dncnn(
        '../datasets/test',
        '../noisy_data/poisson'
    )