from ai_funcs import iCap, predict

pic = iCap()
print(pic.shape)
label = predict(pic)

print(label)