Scrutiny Desk - 20 test pages (all synthetic)

genuine/    10 untouched pages
tampered/   10 pages with one edit each (2 of every kind: replacement, insertion, removal, copy-move, splicing)
tampered/true_masks/   the exact pixels that were changed (white = edited)
answer_key.csv         what was changed on each page, which field, which document type

They come from 40 new applications (APP50000-APP50039) that the model never saw in training,
including a page layout that only appears in testing. Drop any of them into the app's "Tamper lab".
