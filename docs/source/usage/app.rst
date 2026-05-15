The vital_sqi web app
=====================

``vital_sqi`` ships a Dash-based web application that wraps the library's
most common workflows: load a raw recording, run the SQI extraction
pipeline, browse the results segment by segment, tweak the classifier
thresholds, and export the per-segment decisions.

It is implemented in :mod:`vital_sqi.app` and is the recommended way to
explore a recording before committing to a programmatic batch run.

.. contents:: On this page
   :local:
   :depth: 2


Launching the app
-----------------

Development mode (with auto-reload disabled by default — see
`Why is auto-reload off?`_)::

    python -m vital_sqi.app.index

The Dash server starts on http://127.0.0.1:8050 and prints something like::

    INFO:vital_sqi.app.app:Background callbacks enabled; cache at .cache/vital_sqi_app
    Dash is running on http://127.0.0.1:8050/

Press **Ctrl-C** in the terminal to stop.

For a production deployment, import the underlying Flask ``server``
object and run it behind any WSGI host (gunicorn, uvicorn, waitress, …)::

    # wsgi.py
    from vital_sqi.app.app import server

    if __name__ == "__main__":
        from waitress import serve
        serve(server, host="0.0.0.0", port=8050)


Background-callback cache
^^^^^^^^^^^^^^^^^^^^^^^^^

The Compute view's long-running SQI extraction runs as a Dash background
callback backed by `diskcache <https://pypi.org/project/diskcache/>`_.
The cache directory is ``./.cache/vital_sqi_app/`` by default and is
included in ``.gitignore``.  Override with the
``VITAL_SQI_APP_CACHE_DIR`` environment variable if you need a different
location.

If ``diskcache`` is not installed the app still imports and runs, but
the Compute view falls back to a foreground callback (the UI blocks
until the run finishes).  Install it with::

    pip install diskcache


Why is auto-reload off?
^^^^^^^^^^^^^^^^^^^^^^^

Werkzeug's auto-reloader sometimes throws ``OSError [WinError 10038]``
when restarting the dev server on Windows.  We default
``use_reloader=False`` in the ``__main__`` entry point so the log stays
clean.  Re-enable with::

    set VITAL_SQI_APP_RELOAD=1
    python -m vital_sqi.app.index


Sidebar overview
----------------

Five entries in the sidebar, in workflow order:

.. list-table::
   :widths: 12 18 70
   :header-rows: 1

   * - Route
     - Label
     - Purpose
   * - ``/``  ·  ``/views/compute``
     - **Compute**  *(default landing page)*
     - Drop a raw recording, pick wave type / segment params, run
       :func:`~vital_sqi.pipeline.pipeline_functions.extract_sqi`.
       A small card at the bottom also accepts a pre-computed SQI
       table (CSV / JSON) for users skipping the pipeline step.
   * - ``/views/inspect``  *(alias: ``/views/dashboard1``)*
     - **Inspect**
     - Visualise the SQI table with a colour-coded timeline plus a
       per-segment waveform + rule-trace panel.  Threshold mode and the
       set of active SQIs are configurable live.
   * - ``/views/calibrate``
     - **Calibrate**
     - Run the
       :func:`~vital_sqi.calibration.run_calibration.calibrate`
       experiment from the GUI and write the resulting
       ``rule_dict.json`` / ``sqi_dict.json`` as the new bundled
       defaults.
   * - ``/views/export``
     - **Export**
     - Bundle the current run as a decisions CSV, a snapshot rule
       JSON, a ZIP of accepted-segment CSVs, or a standalone HTML
       report.

Inspect and Export start **disabled**; they unlock once an SQI table
is in the ``dataframe`` store — either from a Compute run or from the
SQI-table upload card on Compute.  Compute and Calibrate are always
enabled because they don't depend on a loaded recording — Calibrate
generates its own synthetic signals.


The Compute view
----------------

End-to-end flow:

.. code-block:: text

    +------------------------------------------------------------+
    | [Drag & drop a recording]                                  |
    +------------------------------------------------------------+
    | Wave type: PPG ▾ | fs: auto | duration: 30 s | overlap: 0  |
    +------------------------------------------------------------+
    | Signal column: [pleth — array of 100 samples/row (rec.)] ▾ |
    +------------------------------------------------------------+
    | SQI dictionary: ◉ Bundled  ○ Upload custom                 |
    | Parallel workers: 1                                        |
    +------------------------------------------------------------+
    | [ Run ]  [ Cancel ]   Computing SQIs over 278 segments…    |
    |                       ████████████████░░░░░░░░░  50 %      |
    +------------------------------------------------------------+

Supported recording formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Handled by :mod:`vital_sqi.app.util.waveform_loader`:

**Flat CSV** (one sample per row)
    Columns may include a timestamp (datetime or numeric seconds /
    milliseconds / microseconds — detected from the column name's unit
    suffix) and one or more numeric signal columns.  The loader picks
    a sensible default and shows the others in the *Signal column*
    dropdown.

**Oucru row-per-second CSV**
    The SmartCare / OUCRU exporter writes one row per second; each
    waveform column is a JSON list of samples for that second.  The
    loader detects this automatically by sniffing the first non-empty
    cell of the chosen signal column.  Sampling rate is inferred from
    the array length of the first row (i.e. "100 samples per second
    means 100 Hz").

**EDF / EDF+** (``.edf``, ``.bdf``)
    Delegated to :func:`vital_sqi.data.signal_io.ECG_reader` /
    :func:`~vital_sqi.data.signal_io.PPG_reader`.  Single-channel
    output; the dropdown is hidden.

**MIT-WFDB** (``.hea`` + ``.dat`` / ``.mat``)
    Same delegation as EDF.  Drop the ``.hea`` file; the matching
    sidecar must be uploaded too if your browser doesn't ship it
    automatically (rare with modern Chrome).

Signal-column dropdown
^^^^^^^^^^^^^^^^^^^^^^

As soon as a CSV is staged, :func:`~vital_sqi.app.util.waveform_loader.introspect_columns`
scans every column and ranks the candidates.  Recommended columns
appear first with ``(recommended)`` after their name; for the OUCRU PPG
file you would see::

    pleth     — array of 100 samples/row   (recommended)
    red       — array of 100 samples/row
    ir        — array of 100 samples/row
    perfusion — array of 100 samples/row
    hr        — numeric, 8338 rows
    o2        — numeric, 8338 rows
    …

Picking a different column re-runs the loader against the *same*
upload — no re-drop required.

Choosing the SQI dictionary
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two options:

* **Bundled calibrated default** — points at the
  ``vital_sqi/resource/sqi_dict.json`` shipped with the package; covers
  the full SQI catalogue documented in :doc:`introduction`.
* **Upload custom JSON** — drop a hand-crafted ``sqi_dict.json`` to
  restrict computation to a subset of SQIs or to override per-SQI
  arguments.

What happens on Run
^^^^^^^^^^^^^^^^^^^

The Run callback (background-callback when diskcache is installed)
chains:

1. :func:`~vital_sqi.app.util.waveform_loader.load_from_upload` →
   ``LoadedWaveform``.
2. :func:`~vital_sqi.preprocess.segment_split.split_segment` →
   ``segments``, ``milestones``.
3. :func:`~vital_sqi.pipeline.pipeline_functions.extract_sqi` →
   per-segment SQI DataFrame.
4. Writes both the SQI table and the raw waveform to the app's
   ``dcc.Store`` so the Inspect view can read them.
5. Unlocks the Inspect and Export tabs.

Failures at any step display a one-line error in the status bar and a
full traceback in the server log; the disabled tabs stay disabled.


The Inspect view
----------------

Phase 2 of the app enhancement plan.  Layout::

    ╔════════════════════════════════════════════════════════════════╗
    ║  Recording: rec.csv — PPG @ 100 Hz, 833 800 samples (8338 s)   ║
    ║  278 segments × 49 SQI columns                                 ║
    ╚════════════════════════════════════════════════════════════════╝

    ╔══ Rules ════════════════════════════════════════════════════════╗
    ║  Threshold mode: ● Quantile  ○ Auto-tune  ○ Manual              ║
    ║  Accept band: ░───●────────────────●─── p5–p95                  ║
    ║  ☑ kurtosis_sqi  ☑ perfusion_sqi  ☑ correlogram_sqi             ║
    ║  ☑ msq_sqi       ☑ dtw_sqi                                      ║
    ║  Auto-skipped: zero_crossings_rate_sqi, ectopic_sqi, hfe_sqi, … ║
    ║  [ Drop strictest rule ]   would drop: msq_sqi (rejected 36)    ║
    ╚═════════════════════════════════════════════════════════════════╝

    ╔══ Timeline ════════════════════════════════════════════════════╗
    ║   ▏██▏▏▏██████▏██████████▏▏▏███████████▏▏██████▏▏██████████▏  ║
    ║                                                                ║
    ║                Click a cell to inspect…                        ║
    ╚════════════════════════════════════════════════════════════════╝

    ╔══ Segment 0023 • t=690 s • Δ30.0 s • accept ═══════════════════╗
    ║  ┌──── waveform ────┐  ┌──── SQI values ────┐                  ║
    ║  │                  │  │ kurtosis     3.21  │                  ║
    ║  │ ~~~~~~~~~~~~~~~  │  │ perfusion   12.13  │                  ║
    ║  │                  │  │ correlogram  0.84  │                  ║
    ║  └──────────────────┘  └────────────────────┘                  ║
    ║                                                                ║
    ║  Rule trace                                                    ║
    ║   1. kurtosis_sqi    3.21    ACCEPT                            ║
    ║   2. perfusion_sqi  12.13    ACCEPT                            ║
    ║   3. correlogram_sqi 0.84    ACCEPT                            ║
    ║   4. msq_sqi         0.95    ACCEPT                            ║
    ║   5. dtw_sqi        32.45    ACCEPT                            ║
    ╚════════════════════════════════════════════════════════════════╝


Rule selection panel
^^^^^^^^^^^^^^^^^^^^

Three concepts the user controls:

**Threshold mode**
    See :doc:`pipeline` for the math.  Three options:

    * **Quantile** — symmetric ``(lower, upper)`` trim controlled by
      the range slider.  Default p5/p95.  Tightening rejects more,
      loosening (e.g. p1/p99) accepts more.
    * **Auto-tune** — the per-rule quantile is computed from the
      target slider so the *joint* accept rate hits the target
      (default 85 %) under the independence approximation.  Much more
      forgiving than plain Quantile when several rules are active.
    * **Manual** — the bounds in ``rule_dict.json`` are used verbatim.
      Pick this when applying an externally calibrated rule set.

**Rules checklist**
    Every SQI that has a definition in the active rule dict *and* a
    non-degenerate distribution in this recording appears as a
    toggleable chip.  The default pre-checks a small validated subset:

    .. code-block:: python

        DEFAULT_RULE_COLUMNS = (
            "kurtosis_sqi",
            "perfusion_sqi",
            "correlogram_sqi",
            "msq_sqi",
            "dtw_sqi",
        )

    SQIs whose values are essentially constant across the recording
    (e.g. ``zero_crossings_rate_sqi`` on mean-centred clean PPG, or
    ``ectopic_sqi`` when no ectopics are detected) collapse to a
    zero-width auto-mode band.  They would reject every segment and
    drown out the real quality signal, so they are listed as
    *Auto-skipped* below the checklist and disabled.

**Drop strictest rule**
    A one-click button that flags whichever active rule has an
    upward-outlier rejection count (modified Z-score on per-rule
    rejection tallies, threshold = median + 3·MAD).  When you trust the
    underlying signal but suspect one SQI is too strict for this
    recording, click it to drop the rule and watch the timeline update.


Timeline & detail panel
^^^^^^^^^^^^^^^^^^^^^^^

* The colour-coded bar chart shows every segment in chronological
  order; each cell is one segment, coloured green (``accept``) or red
  (``reject``).  Click a cell to load it into the detail panel.
* The segment-picker dropdown holds the same set, filterable by the
  *all / accept / reject* radios.
* The detail panel plots the segment's raw samples via Plotly (using
  ``Scattergl`` for performance — works up to ~10 M points).  The SQI
  values for that row are shown in a side table.
* The rule trace at the bottom lists every active rule with its value
  and outcome.  The **first** rule to reject is highlighted in red
  with a ``← decisive`` marker so you can see *why* a segment was
  flagged.

Classification runs on demand inside the view: whenever any of the
rule-panel controls or the loaded SQI table change, the timeline and
detail panel update without a page reload.


Robust mode (Phase 4)
^^^^^^^^^^^^^^^^^^^^^

Picking **Robust** in the *Threshold mode* radio replaces the entire
rule-band pipeline with
:func:`~vital_sqi.rule.classify_segments_robust`.  The classifier
ignores the rule dictionary and instead:

1. Rank-normalises every SQI column within the recording.
2. Combines them into a per-segment consensus score in ``[0, 1]``.
3. Detects the recording-level regime (``clean`` / ``bimodal`` /
   ``heavy_noise``) using a bimodality coefficient + GMM check.
4. Applies one of three regime-specific decision functions.

A small badge appears next to the Rules title showing the detected
regime, the mean consensus score, and a ``file flagged`` marker when
fewer than 20 % of segments accept — useful triage for very poor
recordings.

The rule checklist still applies as a *column whitelist*: only the
checked SQIs contribute to the consensus score.  The sliders are
hidden because robust mode has no per-rule trim to set.


The Calibrate view (Phase 3)
----------------------------

Wraps :func:`vital_sqi.calibration.run_calibration.calibrate` in a
background callback so the user can re-derive thresholds without
touching a notebook.  Layout::

    ╔════════════════════════════════════════════════════════════════╗
    ║  Wave type: PPG ▾ | Accept: 200 | Reject: 50 | duration 30 s   ║
    ║  Lower pct: 5  | Upper pct: 95 | Seed: 42                      ║
    ╚════════════════════════════════════════════════════════════════╝
    ║  [ Run calibration ]  [ Cancel ]                               ║
    ║                       █████████████████████████░░░░░ 88 %      ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  Thresholds (latest dry run)                                   ║
    ║   sqi          calibrated  lower   upper  n_accept  …          ║
    ║   kurtosis_sqi    True     0.52    7.14   1850      …          ║
    ║   perfusion_sqi   True    12.30   95.21   1850      …          ║
    ║   …                                                            ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  [ Save as default ]   ✓ Saved 49 rules + 49 SQI templates.    ║
    ║                          Backup: backup_20260518_142331        ║
    ╚════════════════════════════════════════════════════════════════╝

**Run** always invokes ``calibrate(dry_run=True)``.  The thresholds
are stored in a session-only ``dcc.Store`` and displayed as a
sortable / filterable DataTable; uncalibrated SQIs (constant
distribution, all-NaN, etc.) are still shown but italicised in grey
so the user can see exactly what got skipped.

**Save as default** is the only place the app writes files.  It
re-runs the standard exporters
(:func:`~vital_sqi.calibration.exporter.export_rule_dict` /
:func:`~vital_sqi.calibration.exporter.export_sqi_dict`) on the
cached thresholds, which timestamp-back-up the previous defaults
before overwriting ``vital_sqi/resource/rule_dict.json`` and
``vital_sqi/resource/sqi_dict.json``.  The Save button stays disabled
until a successful Run, so the destructive write requires two
deliberate clicks.

The Run button itself is independent of the rest of the app's state —
no recording or SQI table needs to be loaded.  This is the calibration
step that *produces* the bundled rule dict; it doesn't read from any
other view.


The Export view
---------------

Phase 5 replaces the old Rules / Apply pair with a single Export
page.  Each card on the page produces one self-contained artefact
the user can share or feed into a downstream pipeline; no files
touch disk until the corresponding button is pressed.

**Decisions CSV**
    The SQI table with the current accept/reject column from
    Inspect's classifier.  One row per segment; one column per SQI
    plus a final ``decision`` column.

**Rule dict JSON**
    A snapshot of the bounds Inspect is currently applying, in the
    same shape as ``vital_sqi/resource/rule_dict.json``.  Replay
    with::

        from vital_sqi.pipeline.pipeline_functions import classify_segments

        classify_segments(
            sqis,
            rule_dict_filename="rule_dict_snapshot.json",
            ruleset_order=…,
            auto_mode="manual",   # use the snapshot bounds verbatim
        )

    Only available in **Quantile** and **Auto-tune** modes (Manual
    mode would just re-export the bundled defaults; Robust mode has
    no per-rule bounds).

**Accepted segments ZIP**
    Per-segment CSVs of the *raw* waveform for every segment Inspect
    flagged as accept, bundled into a single archive.  Requires the
    raw waveform to be available — i.e. the user must have come
    through Compute rather than uploading a pre-computed SQI table.

**HTML report**
    A standalone single-file report with the recording summary,
    accept/reject counts, per-SQI summary statistics, and the active
    configuration.  Inline CSS, no external dependencies, safe to
    email or commit.


How state flows between views
-----------------------------

State persists in browser stores (``dcc.Store``).  Phase 5 simplified
this map significantly — there's no more ``rule-set-store`` or
``rule-dataframe``; the rule dict always comes from the bundled file.

.. list-table::
   :widths: 28 14 58
   :header-rows: 1

   * - Store ID
     - Lifetime
     - What's in it
   * - ``dataframe``
     - **memory**
     - The current SQI table (one row per segment).  Written by
       Compute (or by the SQI-table upload card on Compute), read by
       Inspect and Export.
   * - ``raw-waveform``
     - **memory**
     - Raw signal samples plus sampling rate / wave type.  Set only
       when Compute runs from a recording.  Required for the Inspect
       view's per-segment waveform panel and for Export's
       "accepted-segments ZIP" download.
   * - ``segment-milestones``
     - **memory**
     - ``{"start": [...], "end": [...]}`` sample indices for each
       segment.  Companion to ``raw-waveform``.
   * - ``inspect-decisions``
     - **memory**
     - The per-segment accept/reject decisions Inspect computes.
       Promoted to the app root in Phase 5 so Export can read the
       same decisions without re-running the classifier.

All stores are memory-only, which is intentional: a 2-hour PPG
recording trivially exceeds the browser's 5 MB ``localStorage``
quota, and the workflow assumes a single linear session (drop a
recording, run Compute, then Inspect + Export immediately).  A page
refresh resets everything — by design.


Troubleshooting
---------------

**Inspect / Export tabs stay disabled.**
    They unlock only after a valid SQI table is in the ``dataframe``
    store.  Run Compute, or use the "Already have an SQI table?" card
    at the bottom of the Compute view.

**"All segments rejected" on an apparently clean recording.**
    The default classifier is Quantile p5/p95 across **five rules**;
    independent trimming compounds.  Switch the threshold mode to
    *Auto-tune* (slider at 85 %) — that's the configuration that gave
    ~80 % accept on our reference SmartCare PPG recording (the same
    file is documented as a worked example in
    :doc:`pipeline`).

**"None-XX.csv" files appearing under the notebook tutorial folder.**
    A notebook-execution artefact; the files are ``save_segment``'s
    output when called with ``segment_name=None``.  Cleaned up
    automatically by the latest tutorials and excluded from version
    control.

**``OSError: [WinError 10038]`` in the server log.**
    Werkzeug auto-reloader closing a dev socket during a restart.
    Harmless; we disable the reloader by default to suppress it
    (see `Why is auto-reload off?`_).

**Browser console errors after a callback.**
    Open the browser dev-tools network tab.  The Python traceback
    appears both in the failed JSON response and in the terminal
    where you launched the app.  Every callback uses
    ``logger.exception`` rather than ``print``, so each line is tagged
    with the module name (``vital_sqi.app.views.<name>``) for grep.


What's coming
-------------

Phase 6 of the :doc:`app enhancement plan
<../../../dev_docs/app_enhancement_plan>`:

* Server-side state (Flask-Caching + per-session keys) so refreshes
  survive and multi-tab use stays isolated.

Phases 0 – 5 are shipped:

* Phase 0 — housekeeping (logging, example downloads, app entry-point
  docs).
* Phase 1 — Compute view + CSV/EDF/WFDB/Oucru loader.
* Phase 2 — Inspect view with timeline, per-segment waveform, and the
  threshold-mode/auto-tune controls.
* Phase 3 — Calibrate view.
* Phase 4 — Robust classifier as a fourth threshold mode in Inspect.
* Phase 5 — Streamlined sidebar (Compute → Inspect → Calibrate →
  Export); replaced the legacy Home / Rules / Apply dashboards with
  a single Export view that produces the decisions CSV, snapshot
  rule JSON, accepted-segment ZIP, and HTML report.


See also
--------

* :doc:`pipeline` — the underlying programmatic workflow exposed by
  the GUI, including the full math behind the threshold-mode options.
* :doc:`../docstring/vital_sqi.rule` — API reference for
  :class:`~vital_sqi.rule.Rule`, :class:`~vital_sqi.rule.RuleSet`,
  and the :mod:`~vital_sqi.rule.auto_threshold` helpers.
* :doc:`../docstring/vital_sqi.pipeline` — API reference for
  :func:`~vital_sqi.pipeline.pipeline_functions.extract_sqi` and
  :func:`~vital_sqi.pipeline.pipeline_functions.classify_segments`.
