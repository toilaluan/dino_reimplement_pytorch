import timm
def main():
    model_name = "vit_base_patch16_384"
    model_official = timm.create_model(model_name, pretrained=True)
    model_official.eval()
    print((model_official))
main()