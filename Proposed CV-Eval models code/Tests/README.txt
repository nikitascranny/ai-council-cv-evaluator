CV-EVALUATOR UNIT TEST SUMMARY

This file lists every unit test in the project and explains what each one checks. There are 40 tests in total, covering 5 parts of the program.

How to run them:
  python -m pytest tests/ -v


1. TESTS FOR SAVING RESULTS TO FILES (test_results_io.py)
-----------------------------------------------------------
These tests check that results get saved to disk correctly and that
file names are cleaned up properly.

  1. Checks that spaces and punctuation in a job title get turned into
     underscores, so the file name is safe to use.
  2. Checks that letters, numbers, and hyphens are left alone (not
     changed) when cleaning up a name.
  3. Checks that several bad characters next to each other only produce
     ONE underscore, not one per character.
  4. Checks that cleaning an empty piece of text just gives back an
     empty result, without crashing.
  5. Checks that saving results automatically creates any missing
     folders needed to store the file.
  6. Checks that the saved file actually contains the correct data.
  7. Checks that the function correctly reports back the full file path
     it just saved to.
  8. Checks that names with special characters (like "Zoë") are saved
     properly and not turned into strange text codes.


2. TESTS FOR TALKING TO THE AI MODEL (test_ollama_utils.py)
-----------------------------------------------------------
These tests check the code that sends prompts to the AI model and
reads its replies, without needing the AI model to actually be running.

  1. Checks that "thinking" text some models add (wrapped in <think>
     tags) gets removed properly.
  2. Checks that this also works when a reply has more than one
     "thinking" section in it.
  3. Checks that "thinking" text spread across multiple lines is still
     removed correctly.
  4. Checks that normal replies with no "thinking" text are left
     unchanged.
  5. Checks that a clean JSON reply from the model is read correctly.
  6. Checks that JSON can still be found even if the model adds extra
     comments or explanation around it.
  7. Checks that the program correctly flags an error if no JSON can be
     found in the model's reply at all.
  8. Checks that a full, successful round-trip (sending a prompt,
     getting a reply, reading the result) works as expected.
  9. Checks that "thinking" text is stripped out BEFORE the reply is
     read as JSON, so it doesn't cause problems.
  10. Checks that the request sent to the model contains the correct
      settings (which model to use, the message, and other options).
  11. Checks that if the connection to the model fails, the program
      reports a clear error instead of crashing.
  12. Checks that if the model gives back a reply that isn't valid
      JSON, the program reports a clear "parse error" instead of
      crashing.


3. TESTS FOR THE ESSENTIAL REQUIREMENTS CHECK (test_essential_requirements.py)
-------------------------------------------------------------------------------
These tests check the part of the program that summarises how well a
candidate matches a job's basic requirements.

  1. Checks that the message sent to the model includes both the job
     description and the candidate's CV details.
  2. Checks that the message sent to the model clearly asks for a JSON
     reply.
  3. Checks that, on success, the candidate's name is correctly added
     to the result.
  4. Checks that the default AI model is used when no other model is
     specified.
  5. Checks that a different model can be used instead, if one is
     specified.
  6. Checks that if the model fails to respond properly, the result
     still includes the candidate's name and a clear error message.
  7. Checks that only the candidate's name and summary are kept when
     preparing results for a later step (extra details are dropped).
  8. Checks that if a summary is missing, it's replaced with an empty
     value instead of causing an error.
  9. Checks that preparing an empty list of results doesn't cause any
     problems.


4. TESTS FOR SCORING CANDIDATES AGAINST A JOB (test_evaluate_cv.py)
-----------------------------------------------------------------------
These tests check the part of the program that scores a candidate's CV
against a job description, and the settings used to run the program.

  1. Checks that, on success, the candidate's name is correctly added
     to their score result.
  2. Checks that the job description and CV text are both correctly
     included in the message sent to the model.
  3. Checks that if scoring fails, the program still returns a safe,
     complete result (with no score, empty lists, and an error message)
     instead of crashing.
  4. Checks that if no extra options are given when running the
     program, the "essential requirements" check is switched off by
     default.
  5. Checks that typing the long option ("--essential-requirements")
     correctly switches that check on.
  6. Checks that typing the short option ("-e") also correctly switches
     that check on.


5. TESTS FOR READING CVs AND JOB DESCRIPTIONS (test_document_utils.py)
--------------------------------------------------------------------------
These tests check the part of the program that pulls text out of PDF
files.

  1. Checks that text from multiple pages of a job description is
     joined together correctly.
  2. Checks that blank pages are safely skipped, rather than causing
     errors or blank sections in the text.
  3. Checks that a job description with only one page is still read
     correctly.
  4. Checks that a PDF where every page is blank simply returns an
     empty result, without crashing.
  5. Checks that a candidate's CV file is correctly passed along to be
     parsed, and that the result comes back as expected.


NOTE ON WHAT ISN'T COVERED
---------------------------
The actual CV-parsing logic (the part that reads and understands the
layout of a CV, in a separate file) is not tested here, since the testing of this feature was done by my teammate.
Everything above works around it
using a stub, so the rest of the program can still be tested
properly on its own.

The main() function that ties everything together and runs the whole
program from start to finish (reading folders, going through every job
and every candidate, and saving all the files) also isn't tested here.
That function is better suited to a bigger, end-to-end test using sample
files, rather than a quick unit test. This will be tested thoroughly in the future.
