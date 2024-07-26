import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from huggingface_hub import login

login(token="")

# Load the base model
base_model_name = "Qwen/Qwen2-1.5B"
adapter_model_name = "tryhighlight/qwen2-1.5B-ocr-task-detection"

config = PeftConfig.from_pretrained(adapter_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
)

# Load the adapter
model = PeftModel.from_pretrained(model, adapter_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

model.eval()  # Set the model to evaluation mode

def apply_chat_template(system_prompt, user_name, ocr_text, max_len):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "My name is " + user_name + " \n" + ocr_text},
    ]
    tokenized_output = tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=False,
                            padding=False,
                            max_length=max_len,
                            truncation=True,
                    )
    input_ids = torch.tensor(tokenized_output, dtype=torch.int)
    print (input_ids.shape)
    return input_ids

def generate_response(system_prompt, user_name, ocr_text):
    max_len = 4096
    input_ids = apply_chat_template(system_prompt, user_name, ocr_text, max_len)
    input_ids = input_ids.to(model.device)
    # Ensure input_ids is 2D
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    # Create an attention mask for the input_ids
    attention_mask = torch.ones_like(input_ids, device=model.device)

    # Check dimensions
    print("Input IDs Shape:", input_ids.shape)
    print("Attention Mask Shape:", attention_mask.shape)

    outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
system_prompt = "You are a helpful AI assistant. User will provide full name followed by the OCR content of his/her computer screen. Looking at the OCR, first detect if there are any email or messaging or any other kind of conversations in it. If yes, then detect if there are any TODOs that the above user has to complete as a result of the conversation. If yes, just provide a short single line task that can be directly added to the todo list. If there is no conversation detected or no task detected in the conversation as a TODO, just output the exact phrase 'No task'."
user_name = "Karthick"
ocr_text = "+9 tem PR\n\n@ FF85m\n\nhefim92.siirradyosu.com\n\nBabA sah mi HAE Se,\n\n \n\nSHE: CHD beat R HE BAM IOFT >\n\nHAGABRERE BARAT\n\n07-2Your Approval is Overdue: Access Request for matt.smith@enron.com\nARSystem <arsystem@mailman.enron.com>  Mon, Dec 31, 2001 at 07:18 PM\nto Allen, Phillip K. <k..allen@enron.com>\nThis request has been pending your approval for  59 days.  Please click http://itcapps.corp.enron.com/srrs/auth/emailLink.asp?ID=000000000067320&Page=Approval to review and act upon this request.\r\n\r\nRequest ID          : 000000000067320\r\nRequest Create Date : 10/11/01 10:24:53 AM\r\nRequested For       : matt.smith@enron.com\r\nResource Name       : Risk Acceptance Forms Local Admin Rights - Permanent\r\nResource Type       : Applications3 04:13:01 fab iF\n\n \n\ndocument.writeIn(' 4] 2b > 4} #1460 wih > 3b > de ASO) sb\n\n \n\nHah RM AK o\n\ndocument.writein(’ #4] ab + 4018 A Wil» 3b > de ADS A] ab AL\n\n \n\nALAG > Md KR ©\nHALE k-NASs RRARALY PY > AMRAMARF > LHR FRBT PR FFU: Fee\nAM ERRR—EM RIES to — 450003 AM EMM RRS > L—- FFARR ET ©\n\n  \n\n \n\nBNAWAWNHM\n\niro)\n\nK\n\n \n\nREDR\nal BE HK\nKLE RR\n\n \n\n \n\n \n\nnah ie\n\n \n\nBanik A\n\n \n\nAP 70007 MAES AoMk\n\nHh eR HB T $30 aE Ht\n\n \n \n\nFIL\" FATE ASF Ba..\n\n \n\nA RA HE LES\n\nHRA RA wr BA\n\n \n\nBEwAP EE.\n\nQBN AN AY RBI NE\n\n \n\nAsks 6 AE\n\nSkea\n\n \n\nApH he\n\f"

response = generate_response(system_prompt, user_name, ocr_text)
print("Generated response:", response)
