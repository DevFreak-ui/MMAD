"""
Script to train a multilingual image captioning model
combining the power of siglip and nllb
"""

from PIL import Image
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer
)

# The tokenization method is `<tokens> <eos> <language code>` for source
# language documents, and `<language code>
# <tokens> <eos>` for target language documents.


class Tokenizer:
    """
    Tokenizer class
    """

    def __init__(self, target_lang: str):
        self.image_processor, self.text_tokenizer = self.__load_from_huggingface(
            target_lang
        )

        self.text_tokenizer.add_bos_token = False
        self.text_tokenizer.add_eos_token = False

    def __call__(self, image: Image.Image, text: str):
        """
        Tokenize the image and text
        """
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values

        inputs = self.text_tokenizer(
            text_target=text,
            return_tensors="pt",
            return_attention_mask=True
        )

        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data

    def detokenize(self, input_ids):
        """
        Detokenize the input_ids
        """
        return self.text_tokenizer.decode(input_ids, skip_special_tokens=False)

    def __load_from_huggingface(self, target_lang="yor_Latn"):
        siglip_image_processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-256-multilingual"
        ).image_processor
        nllb_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            tgt_lang=target_lang
        )
        print(type(nllb_tokenizer))
        return siglip_image_processor, nllb_tokenizer


class SiglipNllb(nn.Module):
    """
    Multilingual Image Captioning Model.
    """

    def __init__(self):
        super().__init__()
        self.vit, self.lm, self.lm_head = self.__load_from_huggingface()
        self.connector = nn.Linear(768, 1024)

    def forward(self, tokens):
        """
        Forward pass
        """
        image_features = self.vit(pixel_values=tokens["pixel_values"])
        image_features = self.connector(image_features.last_hidden_state)

        nllb_output = self.lm(
            tokens["input_ids"],
            tokens["attention_mask"],
            encoder_hidden_states=image_features,
        )
        output = self.lm_head(nllb_output.last_hidden_state)

        return output

    def __load_from_huggingface(self):
        """
        Load the siglip and nllb models from huggingface
        """
        siglip_vit = AutoModel.from_pretrained(
            "google/siglip-base-patch16-256-multilingual"
        ).vision_model
        nllb = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M"
        )
        nllb_decoder = nllb.get_decoder()
        lm_head = nllb.lm_head
        # with open("siglip.txt", "w", encoding="utf-8") as file:
        #     file.write(str(siglip_vit))
        # with open("nllb.txt", "w", encoding="utf-8") as file:
        #     file.write(str(nllb_decoder))
        #     file.write("\n")
        #     file.write(str(lm_head))
        return siglip_vit, nllb_decoder, lm_head 


if __name__ == "__main__":
    tokenizer = Tokenizer("yor_Latn")
    image_ = Image.open("dbd9-200o200o2-hJoHtsRMVfTiv5FVbUimWw.jpg").convert(
        "RGB"
    )
    tokenized_input= tokenizer(
        image_, "Ajá aláwọ̀ búráwùn àti funfun kan ń ṣàn kọjá nínú yìnyín."
    )
    model = SiglipNllb()
    result = model(tokenized_input)
    print("Done")
