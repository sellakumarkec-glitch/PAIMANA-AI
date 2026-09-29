# Branding, emblem and attribution

## What this project does NOT ship

* **The State Emblem of India.** Its use is restricted under the *State Emblem of India
  (Prohibition of Improper Use) Act, 2005*. `frontend/js/common.js` contains a **neutral
  institutional crest placeholder** drawn for this project. Only an authorised Government
  deployment may substitute the official emblem.
* **The official PAIMANA logo, the MoSPI logo, or the Data-for-Development mark.** The SVGs in
  `common.js` are original artwork in a similar visual idiom, not copies.
* **Photography from the portal.** The four hero backgrounds are original vector illustrations
  (`scene()` in `common.js`), so the build is fully self-contained and carries no image rights.

## What it does replicate

The **design language**: the deep-navy utility strip, the white masthead with amber and navy
pill buttons, the sky-blue navigation band with hover dropdowns, the autoplay hero carousel with
arrows and dot indicators, the Ministry-Wise / Sector-Wise tab pattern with the left rail, and
the navy card headers. Layout and colour conventions are not protected the way logos and
photographs are, and replicating them is what makes the tool feel native to PAIMANA users.

## Required notice

Every page footer states: *"Prototype built for the 'AI for Infrastructure Monitoring' problem
statement. Not an official Government of India website."* **Do not remove this** unless the
deployment is actually authorised by MoSPI.

## Fonts

Poppins is loaded from Google Fonts with a full system fallback stack. For an air-gapped or
NIC-hosted deployment, download the Poppins WOFF2 files (SIL Open Font License) into
`frontend/vendor/fonts/` and replace the `<link>` with a local `@font-face` block.
