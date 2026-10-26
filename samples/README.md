# Sample documents

Every document here is **synthetic**: the institutions, people, numbers and stamps are made up by
[`generator/`](../generator), and each page carries a "Synthetic sample, not a real document" footer.
They come from the app's demo applications, so you can open the same cases in the desk.

Each folder has the uploaded PDF (digitally signed with a test certificate, except where noted)
and the scanned page inside it as a JPEG. For pixel edits, `*_edit_mask.png` marks exactly which pixels were changed.

| Folder | What is wrong | How the desk catches it |
| --- | --- | --- |
| `00_genuine_APP00017` | Nothing: all six documents of a clean application | Every check passes; the application is marked verified |
| `01_pixel_edit_aicte_id` | Fee receipt: one digit of the AICTE ID repainted in the image (`1-9245910816` shown as `1-7245910816`) | Tamper model heatmap, plus the ID check against AICTE records |
| `02_pixel_edit_aicte_id` | Approval letter: the first digit of the AICTE ID repainted (`1-…` shown as `3-…`) | Tamper model heatmap, plus the ID check against AICTE records |
| `03_changed_after_signing` | Approval letter: the AICTE ID was covered and retyped **after** the PDF was signed. The signature still verifies | Signature check: the file was changed after signing; the signed and current versions are shown side by side |
| `04_unsigned` | NBA certificate uploaded without a digital signature | Signature check |
| `05_expired_certificate` | NBA accreditation expired before the filing date | Validity date against the filing date |
| `06_name_mismatch` | Institution name on the certificate differs from the AICTE record | Name against the record, with the differing characters marked |
| `07_aicte_id_mismatch` | Affidavit carries a wrong AICTE ID (`…396117` vs `…390117` in the record) | ID against the record |
| `08_lease_too_short` | About 1.7 years of lease left; the longest course runs 4 years | Land rule (Approval Process Handbook cl. 2.1(e)) |
| `09_reused_page` | The approval letter is the same page another institution uploaded earlier (`earlier_upload_APP00062_eoa_letter.pdf`) | Page fingerprint, confirmed pixel by pixel |

Open a PDF in Adobe Reader to see the signature panel: genuine files show a valid (test) signature,
`03_changed_after_signing` shows that the document was modified after signing.
The test certificate's trust root is `app/demo_data/pki/test_ca.pem`; no private keys are included anywhere in this repository.

## `test_pages/`: for the Tamper lab

20 single pages made after training, from 40 new applications the model never saw: 10 with one edit each (two of every kind: replacement, insertion, removal, copy-move, splicing) and 10 untouched.
`answer_key.csv` says what was changed on each page, and `tampered/true_masks/` holds the exact edited pixels. They are preloaded in the app's **Tamper lab** tab; you can also upload any of them there to check them live.
