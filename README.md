# Instruct-Align
High-and-Low Resource Language Alignment via Continual Crosslingual Instruction Tuning

### Research Paper
This work is part of a series of work on LM adaptability to underrepresented & low-resource languages.

Our paper has been accepted in SEALP workshop in AACL 2023. In the meantime, if you use the existing resource, please consider citing:
```
@misc{cahyawijaya2023instructalign,
      title={InstructAlign: High-and-Low Resource Language Alignment via Continual Crosslingual Instruction Tuning}, 
      author={Samuel Cahyawijaya and Holy Lovenia and Tiezheng Yu and Willy Chung and Pascale Fung},
      year={2023},
      eprint={2305.13627},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

If you use the dataset from this work (i.e., NusaX, NusaMenulis, etc) please also consider citing:
```
@inproceedings{winata-etal-2023-nusax,
    title = "{N}usa{X}: Multilingual Parallel Sentiment Dataset for 10 {I}ndonesian Local Languages",
    author = "Winata, Genta Indra  and Aji, Alham Fikri  and Cahyawijaya, Samuel  and Mahendra, Rahmad  and Koto, Fajri  and Romadhony, Ade  and Kurniawan, Kemal  and Moeljadi, David  and Prasojo, Radityo Eko  and Fung, Pascale  and Baldwin, Timothy  and Lau, Jey Han  and Sennrich, Rico  and Ruder, Sebastian",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.57",
    pages = "815--834",
    abstract = "Natural language processing (NLP) has a significant impact on society via technologies such as machine translation and search engines. Despite its success, NLP technology is only widely available for high-resource languages such as English and Chinese, while it remains inaccessible to many languages due to the unavailability of data resources and benchmarks. In this work, we focus on developing resources for languages in Indonesia. Despite being the second most linguistically diverse country, most languages in Indonesia are categorized as endangered and some are even extinct. We develop the first-ever parallel resource for 10 low-resource languages in Indonesia. Our resource includes sentiment and machine translation datasets, and bilingual lexicons. We provide extensive analyses and describe challenges for creating such resources. We hope this work can spark NLP research on Indonesian and other underrepresented languages.",
}

@misc{cahyawijaya2022nusacrowd,
      title={NusaCrowd: Open Source Initiative for Indonesian NLP Resources}, 
      author={Samuel Cahyawijaya and Holy Lovenia and Alham Fikri Aji and Genta Indra Winata and Bryan Wilie and Rahmad Mahendra and Christian Wibisono and Ade Romadhony and Karissa Vincentio and Fajri Koto and Jennifer Santoso and David Moeljadi and Cahya Wirawan and Frederikus Hudi and Ivan Halim Parmonangan and Ika Alfina and Muhammad Satrio Wicaksono and Ilham Firdausi Putra and Samsul Rahmadani and Yulianti Oenang and Ali Akbar Septiandri and James Jaya and Kaustubh D. Dhole and Arie Ardiyanti Suryani and Rifki Afina Putri and Dan Su and Keith Stevens and Made Nindyatama Nityasya and Muhammad Farid Adilazuarda and Ryan Ignatius and Ryandito Diandaru and Tiezheng Yu and Vito Ghifari and Wenliang Dai and Yan Xu and Dyah Damapuspita and Cuk Tho and Ichwanul Muslim Karo Karo and Tirana Noor Fatyanosa and Ziwei Ji and Pascale Fung and Graham Neubig and Timothy Baldwin and Sebastian Ruder and Herry Sujaini and Sakriani Sakti and Ayu Purwarianti},
      year={2022},
      eprint={2212.09648},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Files Structure
- run_t2t_finetuning.py &rarr; main python script for running the cross-lingual alignment training
- run_t2t_finetuning.sh &rarr; shell script for running various cross-lingual alignment training experiments
- nlu_prompt.py &rarr; python script storing the prompts that are used for zero-shot inference prompting
- main_nlu_prompt.py &rarr; main python script for running the zero-shot inference prompting inference
- run_nlu_prompt.sh &rarr; shell script for running zero-shot inference prompting inference using various prompt templates
- run_nlu_exp.sh &rarr; shell script for running zero-shot inference prompting inference for various models
- prompt_utils.py &rarr; utility scripts for prompting
- data_utils.py &rarr; utility scripts for data loading in instruct-align
- augmentation_utils.py &rarr; utility script for constructing instruction data
- notebooks &rarr; contains all notebooks used for analysis

### License
InstructAlign is licensed under the Apache 2.0 license, as found in the [LICENSE](LICENSE) file.
