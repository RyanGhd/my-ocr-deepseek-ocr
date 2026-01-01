# gcs command to copy files
```bash
gsutil -m cp -r ~/images_output/* gs://gcp-wow-wiq-014-test-paa-input/images_output/
```

# Check what's using GPU memory
```bash
nvidia-smi
```

# docling using granite docling 
ref: https://huggingface.co/ibm-granite/granite-docling-258M

```bash
xyz@DESKTOP-2MT MINGW64 ~/Documents/Files/mp/test/my-ocr-deepseek-ocr/src/images/converted (main)
$ docling --to html --to md --pipeline vlm --vlm-model granite_docling "9.png"
```

### valid values for `--to` :
```
 'md', 'json', 'yaml', 'html', 'html_split_page', 'text', 'doctags'.                  
```