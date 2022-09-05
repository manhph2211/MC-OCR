def flip_image(img_path, model):
  transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize((128,128)),
                                transforms.Normalize((0.5,), (0.5,))])
  img = cv2.imread(img_path)
  img1 = transform(img)
  img1 = torch.unsqueeze(img1, dim=0)
  output = model.to('cpu')(img1)
  output = F.log_softmax(output, dim=1) # log softmax
  pred = output.argmax(dim=1, keepdim=True) # argmax, chú ý keepdim, dim=1
  # print(pred)
  if pred[0][0].detach().numpy() == 0:
    img = cv2.flip(img, 0)
    # plt.imshow(img)
    # plt.show()
    cv2.imwrite(img_path, img)
    