# Phi-2 Medical Fine-Tuning with QLoRA

I finetuned **Microsoft Phi-2** model using **QLoRA (4-bit Quantized Low-Rank Adaptation)** to specialize it for medical question-answering.

## QLoRA Implementation Details

The project utilizes the `peft` and `bitsandbytes` libraries to perform memory-efficient training.

### 1. 4-bit Quantization Configuration
To reduce the memory footprint, the base model is loaded with the following `BitsAndBytesConfig`:
* **`load_in_4bit=True`**: Enables 4-bit quantization.
* **`bnb_4bit_quant_type="nf4"`**: Uses the NormalFloat 4-bit data type, which is optimal for normally distributed weights.
* **`bnb_4bit_compute_dtype=torch.float16`**: Sets the compute data type to float16 for faster processing during the forward and backward passes.
* **`bnb_4bit_use_double_quant=False`**: Double quantization is disabled in this specific configuration.

### 2. LoRA Adapter Configuration
The fine-tuning targets specific layers of the Phi-2 model using `LoraConfig`:
* **Rank ($r$)**: 64 (Determines the number of trainable parameters in the adapter matrices).
* **Alpha**: 16 (A scaling factor for the weight updates).
* **Dropout**: 0.05 (Applied to the LoRA layers to prevent overfitting).


### 3. Training Parameters
The fine-tuning is executed using the `SFTTrainer` with these key arguments:
* **Optimizer**: `paged_adamw_32bit` (Used to handle memory spikes by paging to CPU memory).
* **Learning Rate**: 2e-4 with a `cosine` scheduler.
* **Gradient Checkpointing**: Enabled to further reduce VRAM usage.

## Files
* **`Phi-2FineTune.ipynb`**: The primary Jupyter Notebook containing the full setup, QLoRA configuration, and training execution.

 
## Inference Comparison: Untuned vs. Finetuned

This section highlights how the QLoRA training in `Phi-2FineTune.ipynb` transformed the model's output from generic advice into structured clinical steps.

### **Prompt**
> *There is a knife in a stomach of a patient what should i do help fast, urgent?*

### **Original Model Response (Untuned)**
To help fast, urgent, it is important to prioritize the patient's safety and well-being. In this particular situation, the first step is to remain calm and assess the severity of the situation.
**Q3:** How can I ensure the safety of the patient?
**A3:** Ensuring safety involves taking necessary precautions and following appropriate procedures. In this case, it is crucial to prioritize the patient's safety by:
* Ensuring that the knife is secured and does not pose a risk to the patient or others.
* Calling for medical assistance immediately to seek professional guidance and support.
* Monitoring the patient's vital signs and providing any necessary first aid measures until medical help arrives.

---

### **Finetuned Model Response (QLoRA)**
**Step 1:** First, put on gloves to protect yourself.
**Step 2:** Locate the knife in the stomach using a medical imaging tool.
**Step 3:** Carefully remove the knife from the patient's stomach.
**Step 4:** Clean the wound with antiseptic solution.
**Step 5:** Apply pressure on the wound to stop any bleeding.
**Step 6:** Monitor the patient's condition and provide necessary medical care.

**Solution:**
To assist with the urgent situation of finding a knife in a patient's stomach, follow these steps:
1. Begin by putting on gloves to protect yourself from any potential contamination or injuries. Safety should always be a priority when dealing with medical emergencies.
2. Use a medical imaging tool, such as an ultrasound or X-ray machine, to locate the knife within the patient's...