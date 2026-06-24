## Research text writing guidence

When writing the paper think about the story of a paper in the following structure:
- Importance: What are we looking at and why is it relevant? This sets up the broader context and motivation.
- Gap: What is missing in the current literature and what are the limitations of existing work?
- Objective: What specific problem are we trying to solve or question are we trying to answer?
- Method: How do we approach solving this problem? What are our key technical contributions?
- Findings: What are the main experimental results and insights we discovered?
- Implications: What do our findings mean for the field? What are the key takeaways?
- (Optional)Future work: What interesting directions remain to be explored?

## Notes

Disclaimer:
The points below summarise some issues that I find myself often pointing out when reviewing paper drafts.
Don’t treat these points as hard constraints, but rules of thumb; there’s exceptions for almost every rule of course!
The advice below focuses on conventions I’ve seen commonly used in the ML community — other academic disciplines may have their own expectations and standards.

General advice on structure & writing:
 Think about a compelling narrative.  Don’t only state what you do, but why. Motivate solutions you’re presenting by clearly explaining why existing techniques fail.  Readers should be excited to read on! Of course keep in mind that you’re writing an academic paper, not a suspense novel 🙂 But also not a phone book (if these still exist)...
To this end, it may help first starting with an outline with bullet points.  Think carefully about what is introduced where. We can iterate on this outline to avoid rewriting larger parts of the paper.
A helpful exercise can be to think about how every paragraph in the paper motivates (necessitates?) the next one.  If it doesn’t, perhaps there’s a gap that needs to be filled.
It is usually a good idea to have a clear transition between reviewing important background and presenting your contributions. E.g., have a section on “Background and Problem Statement”, which recalls important concepts, so that in the next section you can introduce your algorithmic contributions.
Make sure you can get to discussing your contributions early (i.e., don’t ask readers to read half the paper before your own work starts).
Think about using a running example to motivate your research; thread it through the paper (e.g., use for motivation in the intro; to ground the formal notation in the technical exposition; and ideally return presenting actual experimental results on it)
 Aim to have a complete draft at least two weeks before the deadline… This way, there is time to iterate.
Get feedback from fellow group members, and participate in (and contribute to) the institute-wide paper sharing.  This will help get a diversity of opinions, and catch issues early that might come up during review.
 Put yourself into the perspective of possible reviewers.  What could be possibly misunderstood?  What could be criticised?  What backgrounds might your reviewers have?
Take a look at some papers you really like (and perhaps at some of the best-received papers at conferences, such as best-paper winners, orals etc.)  What elements specifically do you like?  Which elements don’t you like?  Of course don’t overfit...

Title & abstract:
Those are obviously crucial for the reader’s first impression!
It is often a good idea to generate multiple different versions of the title (and possibly of the abstract as well)
More and more conferences have separate abstract deadlines; make sure to have the title & abstract ready early on so we can iterate.

Introduction:
Clearly motivate the problem you’re addressing; why is it important? Why does existing work not sufficiently address it?  Include key citations, but see point below about avoiding repetitions with related work section.
Make clear what the key contributions are (e.g., concise bulleted list at end of introduction section)

Related work:
Be generous citing related work (as long as it actually is related and relevant...)
Think about clustering related work into different conceptually related subsections
Avoid repetitions, e.g., with introduction
Try to use original citations where possible (i.e., not only recent surveys or papers published at last NeurIPS…)
Not only list related work, but make clear how it is related, and why it doesn’t solve the problem you’re considering
Related work should be in present tense (i.e., "XYZ (1730) introduce an algorithm for …”)

Technical exposition / presentation of algorithms, theory etc.
Use clear notation; formally introduce the used notation, especially if non-standard; use it consistently! (Macros can help).  Look at related papers (in similar venues) and stick to common conventions where it makes sense.
Give clear, formal examples (ideally connected to a running example as discussed above)
Sometimes it can help the exposition by starting with important special cases, to introduce important ideas. The most general & abstract setting is often difficult to explain
Don’t try to explain too many different ideas in a single paper — focus on what is really important.  Sometimes less is more.
Present pseudo-code for algorithms. Make sure notation is consistent.
Avoid having ten theorems.  Typically, there’s one (or two) main Theorem(s), and the rest are Corollaries, Lemmas or Propositions.
Discuss limitations and directions for future research, but make clear why they are out of scope, and don’t appear defensive (i.e., avoid suggesting to reviewers possible complaints about why you didn’t do certain things).

Experimental results:
 State explicitly which questions you’re investigating in your experiments
 It can help having a subsection on “Experimental Setup”, which discusses baseline approaches, datasets used, evaluation metrics etc.”, followed by “Experimental Results” discussing the actual numerical results.
Don’t just present the results, but provide your interpretation. Explicitly say what are the most important findings (don’t expect the readers to figure out for themselves from tables etc.)

Figures & tables:
Try to design a figure illustrating key ideas about the problem, approach etc., and include it early on. Perhaps sketch by hand to explore ideas
Use clear captions; present most important take-away messages in caption.
Imagine a reviewer skimming through the figures before actually reading the paper. Make sure they get the most important ideas in this first quick pass.
Assemble figures in page-wide panels (e.g., 3 smaller figures in a row), e.g., using \begin{figure*}, rather than using inline figures. This saves space, and preserves the flow when reading.
Make sure axes are labeled and legible
Use clear legends
Use different line-styles and markers to make sure paper is readable when printed B/W, or for color-blind readers.  Make sure your algorithm sticks out for sake of comparison (e.g., bolder line, etc.)
Use error bars!  State what the randomness is over (e.g., randomness in the algorithm, different data splits, …)
Try to provide details to allow reproducing the experiments (typically in the appendix)
Generally prefer figures over tables (or include tables for completeness in an appendix, and a figure summarising them in the main text)
When using tables, make sure they are easy to interpret (e.g., highlight methods that perform best in your metric with statistical significance, etc.)

LaTeX, language & formatting
For every sentence, ask yourself if you can say the same thing shorter (e.g., “In this paper, we develop learning algorithms that are robust for the task of …”-> “We develop robust learning algorithms for …”)
Use citations correctly:  E.g., avoid using citation numbers as nouns “[1] argue that”.  Rather “Meier et al (2032) argue that”.  E.g., via \citet vs \citep, \citeauthor, \citeyear
Use automatic spell-checking and grammar-checking tools
Prefer active over passive voice
Avoid colloquial language (in particular also “we’ll, aren’t” etc.)
Either use "consistent capitalisation of section titles" or "Consistent Capitalisation of Section Titles”, but commit to the chosen variant (sometimes, formatting guidelines of the conferences/journals specify a default).
Don’t have section titles immediately followed by subsection titles (i.e., “3. Algorithm \\ 3.1 Algorithm Overview”)
Capitalize references to “Section 1”, “Algorithm 2”, “Equation (3)”, but say “In this section, we"
Equations are references with parentheses (using \eqref{})
Make sure acronyms are defined the first time you use them
Use a special font when referring to algorithm names, data sets etc. (e.g., \textsc{})
Carefully use labeled equations (e.g., \begin{equation} vs \begin{equation*} / $$).  Don’t introduce labels you don’t reference later.
In ML venues, it’s very rare to see labeled remarks (e.g., “Remark 1. Our algorithm is the best.”)
Use commas before and after “e.g., i.e.,”, but not after “cf."
Highlight important concepts (e.g., technical terms that appear for the first time) with \emph{}
Qualify findings you consider surprising with “perhaps surprisingly"
Use LaTeX macros for important notation, so it’s easier to do a global replace later. The package xspace (\xspace) comes in handy to handle white spaces correctly.
When relegating proofs of statements to an appendix, make sure to say so in the main body of the text. Also make sure references to the appendix don’t break when compiling the short version of your paper.
Avoid “orphans” (single words on lines), e.g., using \looseness -1
Use suggestive repo/overleaf names (e.g., ICML 2021 PaperTitle)…. Also for file names, please avoid using “main.tex” (which will be compiled to main.pdf…)
