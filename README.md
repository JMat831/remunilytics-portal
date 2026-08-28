# Remunilytics Prospect Portal — deploy repo

This is a generated, minimal copy of just the files needed to run the
Streamlit prospect portal. It is rebuilt from the main working repo by
`portal/build_deploy_repo.py` — do not hand-edit source files here; edit
them in the main repo and re-run that script.

## Deploying

This repo is PUBLIC (Streamlit Community Cloud's free tier requires it).
Nothing sensitive is committed here — the access tokens live only in the
deployed app's private Secrets, never in this repo.

1. On https://share.streamlit.io, "Deploy a public app from GitHub" using
   this repo, branch `master`, main file path `portal/app.py`.
2. In the app's Settings -> Secrets, paste the `tokens_json = '''...'''`
   TOML block printed by `build_deploy_repo.py` (or regenerate it any
   time with `python portal/print_secrets_toml.py` in the main repo).
3. After the first successful deploy, note the app's URL and update
   `PORTAL_BASE_URL` in `portal/generate_tokens.py` in the MAIN repo,
   then re-run `--mint`/`--export` there to produce working links.

## Rebuilding this repo after portal changes

From the main Remunilytics directory:

    python portal/build_pdf_subset.py     # refresh portal/pdfs/ if tokens changed
    python portal/build_deploy_repo.py    # re-sync this folder
    cd ../remunilytics-portal-deploy
    git add -A && git commit -m "Sync from main repo" && git push
