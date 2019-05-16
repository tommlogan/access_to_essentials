% ==============================================================
% RISA Journal Document Class File
% v1.0 for LaTeX 2e
%
% Copyright (C) 2009 Risk Analysis
% for LaTeX version 2E, by Aptara, India
%
% Preamble to this style file
%
% ==============================================================
%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{risa}[2009/05/12 v2.2 LaTeX2e RISA document class]
%
\def\@version{1.0}
\typeout{Document Style `Risa' v\@version, released 2009 May 12}
%
%  Journals use two-sided printing.
%
\@twosidetrue
\@twocolumntrue
\@mparswitchtrue
\def\ds@draft{\overfullrule 5pt}
\def\ds@twocolumn{\@twocolumntrue}
\def\ds@onecolumn{\@twocolumnfalse}

% letters option
%
\newif\if@letters
\def\ds@letters{\@letterstrue}
% landscape option
%
\newif\ifSFB@landscape
\def\ds@landscape{\SFB@landscapetrue}
% galley option
%
\newif\ifSFB@galley
\def\ds@galley{\SFB@galleytrue}
%
\newif\if@usedcolumn
\def\ds@usedcolumn{\@usedcolumntrue}
%
\newif\if@usegraphicx
\def\ds@usegraphicx{\@usegraphicxtrue\usepackage[dvips]{graphicx}}
%
\newif\ifSFB@referee
\def\ds@referee{%
 \SFB@refereetrue
 \@twocolumnfalse
}
\newif\ifdoublesp@cing
\def\ds@doublespacing{%
 \doublesp@cingtrue
}%
%
\newif\ifNHT@useAMS
\def\ds@useAMS{\NHT@useAMStrue}
%
\ExecuteOptions{letterpaper,twoside}
\ProcessOptions
%
%  ****************************************
%  *               FONTS                  *
%  ****************************************
%
\lineskip 1pt
\normallineskip 1pt
\ifSFB@referee
  \def\baselinestretch{1.8}%1.5}
\else
  \ifdoublesp@cing
  \def\baselinestretch{2}
\else
  \def\baselinestretch{1}
\fi\fi
\if@twocolumn
  \renewcommand\normalsize{\@setfontsize\normalsize\@xpt{12}%
   \abovedisplayskip 6pt plus 2pt minus 2pt
   \belowdisplayskip \abovedisplayskip
   \abovedisplayshortskip 6pt plus 2pt
   \belowdisplayshortskip \abovedisplayshortskip
   \let\@listi\@listI}
 \else
 \ifSFB@referee
  \renewcommand\normalsize{\@setfontsize\normalsize\@xiipt{14.5}% From size12.clo
   \abovedisplayskip 4pt plus 1pt minus 1pt
   \belowdisplayskip \abovedisplayskip
   \abovedisplayshortskip 4pt plus 1pt
   \belowdisplayshortskip \abovedisplayshortskip
   \let\@listi\@listI}
 \else
 \renewcommand\normalsize{\@setfontsize\normalsize\@ixpt{11}%
   \abovedisplayskip 4pt plus 1pt minus 1pt
   \belowdisplayskip \abovedisplayskip
   \abovedisplayshortskip 4pt plus 1pt
   \belowdisplayshortskip \abovedisplayshortskip
   \let\@listi\@listI}
 \fi
\fi
\newcommand\small{\@setfontsize\small\@ixpt{11}%
 \abovedisplayskip 4pt plus 1pt minus 1pt
 \belowdisplayskip \abovedisplayskip
 \abovedisplayshortskip 4pt plus 1pt
 \belowdisplayshortskip \abovedisplayshortskip
 \def\@listi{\leftmargin\leftmargini
  \topsep 2pt plus 1pt minus 1pt
  \parsep \z@
  \itemsep 2pt}}
\newcommand\footnotesize{\@setfontsize\footnotesize\@viiipt{10}%
 \abovedisplayskip 4pt plus 1pt minus 1pt
 \belowdisplayskip \abovedisplayskip
 \abovedisplayshortskip 4pt plus 1pt
 \belowdisplayshortskip \abovedisplayshortskip
 \renewcommand\@listi{\leftmargin\leftmargini
  \topsep 2pt plus 1pt minus 1pt
  \parsep \z@
  \itemsep 2pt}}
\newcommand\scriptsize{\@setfontsize\scriptsize\@vipt{8}}
\newcommand\tiny{\@setfontsize\tiny\@vpt{6}}
%
\if@twocolumn
   \newcommand\large{\@setfontsize\large\@xpt{11}}
 \else
  \newcommand\large{\@setfontsize\large\@xpt{12}}
 \fi
\def\@xviiipt{18}
\newcommand\Large{\@setfontsize\Large\@xiipt{14}}
\newcommand\LARGE{\@setfontsize\LARGE\@xivpt{17}}
\newcommand\huge{\@setfontsize\huge\@xviiipt{20}}
\newcommand\Huge{\@setfontsize\Huge\@xxpt{25}}
%
\newcommand\sevensize{\@setfontsize\sevensize\@viipt{8}}
%
\normalsize

%%%%%%%%%%%%%%%%%%%%%%%% Extra Math / Symbol code %%%%%%%%%%%%%%%%%%%%%%%%

  \newcommand{\rmn}[1]{{\mathrm{#1}}}
  \newcommand{\itl}[1]{{\mathit{#1}}}
  \newcommand{\bld}[1]{{\mathbf{#1}}}
%%%
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\newcommand{\mit}{\protect\pmit}
\newcommand{\pmit}{\@fontswitch{\relax}{\mathnormal}}
%

\def\hexnumber#1{\ifcase#1 0\or1\or2\or3\or4\or5\or6\or7%
  \or8\or9\or A\or B\or C\or D\or E\or F\fi}

\def\setboxz@h{\setbox\z@\hbox}
\def\wdz@{\wd\z@}
\def\boxz@{\box\z@}
\def\setbox@ne{\setbox\@ne}
\def\wd@ne{\wd\@ne}

\def\math@atom#1#2{%
   \binrel@{#1}\binrel@@{#2}}
\def\binrel@#1{\setboxz@h{\thinmuskip0mu
  \medmuskip\m@ne mu\thickmuskip\@ne mu$#1\m@th$}%
 \setbox@ne\hbox{\thinmuskip0mu\medmuskip\m@ne mu\thickmuskip
  \@ne mu${}#1{}\m@th$}%
 \setbox\tw@\hbox{\hskip\wd@ne\hskip-\wdz@}}
\def\binrel@@#1{\ifdim\wd2<\z@\mathbin{#1}\else\ifdim\wd\tw@>\z@
 \mathrel{#1}\else{#1}\fi\fi}

% These must be here because \newif and \newtoks are defined \outer.
\newif\iffirstta
\newtoks\math@groups
\newif\ifd@f
% Make these do nothing (for backwards compatibility}
\let\loadboldmathitalic=\relax
\let\loadboldgreek=\relax
% font definitions for NFSS release 2
\newcommand{\textbfit}[1]{\textbf{\textit{#1}}}
\newcommand{\mathbfit}[1]{\textbf{\textit{#1}}}
\newcommand{\textbfss}[1]{\textbf{\textsf{#1}}}
\newcommand{\mathbfss}[1]{\textbf{\textsf{#1}}}
  %
  \def\mathch{\protect\p@mathch}
  \def\p@mathch#1#2{%
    \begingroup
    \let\@nomath\@gobble \mathversion{#1}%
    \math@atom{#2}{%
    \mathchoice%
      {\hbox{$\m@th\displaystyle#2$}}%
      {\hbox{$\m@th\textstyle#2$}}%
      {\hbox{$\m@th\scriptstyle#2$}}%
      {\hbox{$\m@th\scriptscriptstyle#2$}}}%
  \endgroup}
  %
  \def\bmath{\protect\p@boldm}
  \def\p@boldm#1{\mathch{bold}{#1}}
%
  \let\mit=\mathnormal
  %
  % The following is needed because NFSS release 2
  % does not define the bold
  % math symbol font to be available!
  %
%
  \SetSymbolFont{symbols}{bold}{OMS}{cmsy}{b}{n}
\newcommand{\romn}[1] {{\mathrm #1}}
\newcommand\fd{\hbox{$.\!\!^{\reset@font\romn d}$}}
\newcommand\fh{\hbox{$.\!\!^{\reset@font\romn h}$}}
\newcommand\fm{\hbox{$.\!\!^{\reset@font\romn m}$}}
\newcommand\fs{\hbox{$.\!\!^{\reset@font\romn s}$}}
\newcommand\fdg{\hbox{$.\!\!^\circ$}}
\newcommand\farcm{\hbox{$.\mkern-4mu^\prime$}}
\newcommand\farcs{\hbox{$.\!\!^{\prime\prime}$}}
\newcommand\fp{\hbox{$.\!\!^{\reset@font\reset@font\scriptscriptstyle\romn p}$}}
\newcommand\arcmin{\hbox{$^\prime$}}
\newcommand\arcsec{\hbox{$^{\prime\prime}$}}
\def\micron{\hbox{$\mu$m}}
%%%Command to produce a box round selected text
%%%NB This is for use in the file mnguide only -- it is NOT
%%%    to be used in the journal
\newcommand{\boxit}[1]{%
 \noindent\fbox{%
   \begin{minipage}[]{84mm}#1\end{minipage}
 }
}
%%%%%%%%%%%%%%%%%%%%% End of Extra Math / Symbol code %%%%%%%%%%%%%%%%%%%%%

%  ****************************************
%  *            PAGE LAYOUT               *
%  ****************************************
%% %
\ifSFB@landscape
 \setlength\textwidth{\textheight}
 \setlength\textheight{\SFB@measure}
\fi
\ifSFB@referee
%%%% marie modification to get max width with 1 in margins
% \setlength\textwidth{36pc}
 \setlength\textwidth{6.5in}  
\fi
% FOOTNOTES:
\setlength\footnotesep{4pt}
\setlength\skip{\footins 19.5pt plus 12pt minus 1pt}
% FLOATS:
\setlength\floatsep{ 12pt plus 5.5pt minus 1pt}
\setlength\textfloatsep{ 12pt plus 7.5pt}
\setlength\intextsep{ 12pt plus 5.5pt minus 2pt}
\setlength\dblfloatsep{ 12pt plus 5.5pt minus 2pt }
\setlength\dbltextfloatsep{ 15pt plus 4.5pt minus 3pt}
\setlength\@fptop{ 0pt plus 0fil}
\setlength\@fpsep{ 12pt plus 0fil }
\setlength\@fpbot{ 0pt plus 0fil }
\setlength\@dblfptop{ 0pt plus 0fil}
\setlength\@dblfpsep{ 12pt plus 0fil}
\setlength\@dblfpbot{ 0pt plus 0fil}
\setlength\marginparpush{ 5pt }

%  ****************************************
%  *           PARAGRAPHING               *
%  ****************************************
%
\newlength{\realparindent} % needed for abstract
\setlength\parskip{ \z@}% plus .1pt }
\setlength\parindent{ 10pt }
\setlength{\realparindent}{\parindent}
\setlength\partopsep{ 0pt plus 1pt }
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\@beginparpenalty -\@lowpenalty
\@endparpenalty -\@lowpenalty
\@itempenalty -\@lowpenalty
\hyphenpenalty=500
\clubpenalty=0
\widowpenalty=10000

%  ****************************************
%  *       PARTS & SECTIONS NUMBERS       *
%  ****************************************
%
\newcounter {part}
\newcounter {section}
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\def\thepart {\arabic{part}}
\def\thesection {\arabic{section}}
\def\thesubsection    {\thesection.\arabic{subsection}}
\def\thesubsubsection {\thesubsection.\arabic{subsubsection}}
\def\theparagraph     {\thesubsubsection.\arabic{paragraph}}
\def\thesubparagraph  {\theparagraph.\arabic{subparagraph}}

%%%Additional material below is from GJI class file
% Sometimes people want to number equations by section, rather than
% sequentially through the whole paper.

\newcommand\eqsecnum{
    \@newctr{equation}[section]
   \def\theequation{\normalsize\mbox{\arabic{section}.\arabic{equation}}}}

\newcommand\eqsubsecnum{
    \@newctr{equation}[subsection]
    \def\theequation{\normalsize%
    \mbox{\arabic{section}.\arabic{subsection}.\arabic{equation}}}}

%  ****************************************
%  *                PARTS                 *
%  ****************************************
%
\def\part{\par \addvspace{4ex}\@afterindentfalse
 \secdef\@part\@spart}
\def\@part[#1]#2{\ifnum \c@secnumdepth >\m@ne
  \refstepcounter{part}
  \addcontentsline{toc}{part}{Part \thepart: #1}
 \else \addcontentsline{toc}{part}{#1}
 \fi
 {\parindent 0pt \raggedright
  \ifnum \c@secnumdepth >\m@ne
   \reset@font\large\rm PART
   \ifcase\thepart \or ONE \or TWO \or THREE \or FOUR \or FIVE
    \or SIX \or SEVEN \or EIGHT \or NINE \or TEN \else \fi
   \par \nobreak
  \fi
  \reset@font\LARGE \rm #2 \markboth{}{}\par }
 \nobreak \vskip 3ex \@afterheading}
\def\@spart#1{{\parindent 0pt \raggedright
  \reset@font\LARGE \rm #1\par}
 \nobreak \vskip 3ex \@afterheading}

%  ****************************************
%  *               SECTIONS               *
%  ****************************************
%
% \@startsection modified to leave less space before \subsection
%    if it immediately follows \section

% \@startsection {NAME}{LEVEL}{INDENT}{BEFORESKIP}{AFTERSKIP}{STYLE}
\newif\if@levelone
\def\@startsection#1#2#3#4#5#6{\if@noskipsec \leavevmode \fi
   \par \@tempskipa #4\relax
   \@afterindenttrue
   \ifnum #2=\@ne  % if level=1 ie. section
     \global\@levelonetrue
   \else
     \global\@levelonefalse
   \fi
   \ifdim \@tempskipa <\z@ \@tempskipa -\@tempskipa \@afterindentfalse\fi
   \if@nobreak
     \everypar{}\ifnum#2=2 \vskip 0pt plus1pt\fi % was 6pt
   \else
     \addpenalty{\@secpenalty}\addvspace{\@tempskipa}
   \fi \@ifstar
     {\@ssect{#3}{#4}{#5}{#6}}{\@dblarg{\@sect{#1}{#2}{#3}{#4}{#5}{#6}}}}
%
\def\SFB@hangraggedright{\rightskip=\@flushglue \let\\=\@centercr \parindent=0pt}
%
\newif\if@firstsection
\@firstsectiontrue
\def\section{%\fixfootnotes%
 \if@firstsection \@firstsectionfalse \fi%
 \@startsection{section}{1}{\z@}
 {18pt plus 2pt minus 2pt}{\ifbackmatter5.5pt\else6pt\fi}
 {\SFB@hangraggedright\reset@font\normalsize\ifbackmatter\raggedright\bfseries\else\bfseries\raggedright\fi}}
\def\subsection{\@startsection{subsection}{2}{\z@}
 {18pt plus 2pt minus 2pt}{6pt}
 {\SFB@hangraggedright\reset@font\normalsize\bfseries\ifbackmatter\centering\else\raggedright\fi}}
\def\subsubsection{\@startsection{subsubsection}{3}{\z@}
 {18pt plus 2pt minus 2pt}{6pt}
 {\SFB@hangraggedright\reset@font\normalsize\it}}
\def\paragraph{\@startsection{paragraph}{4}{\parindent}
 {0pt}{-0.01em}{\reset@font\normalsize\it}}
%
\def\subparagraph{\@startsection{subparagraph}{5}{\parindent}
 {12pt plus 2.25pt minus 1pt}{-0.5em}{\reset@font\normalsize\it}}
\setcounter{secnumdepth}{3} % SFB 0.1.01
%
% \@sect{NAME}{LEVEL}{INDENT}{BEFORESKIP}{AFTERSKIP}{STYLE}[ARG1]{ARG2}
\def\@sect#1#2#3#4#5#6[#7]#8{%
  \ifnum #2>\c@secnumdepth
    \def\@svsec{}%
  \else
    \refstepcounter{#1}%
    \if@levelone % is it a \section or \section*
      \ifSFB@appendix
        \edef\@svsec{}%
      \else
        \edef\@svsec{\csname the#1\endcsname.\hskip .5em}%
      \fi
    \else
	  \ifnum#2=1
         \edef\@svsec{\csname the#1\endcsname.\hskip .5em}%
	  \else
         \edef\@svsec{\csname the#1\endcsname\hskip .5em}%
	  \fi
    \fi
  \fi
  \@tempskipa #5\relax
  \ifdim \@tempskipa>\z@
    \begingroup #6\relax
    \if@levelone
      \ifSFB@appendix
        \@hangfrom{\hskip #3\relax\@svsec}{\interlinepenalty \@M
        APPENDIX\uppercase{#8}% \csname the#1\endcsname\ifx#8\empty\else:\hskip 0.5em{#8}\fi 
		\par}%
      \else
        \@hangfrom{\hskip #3\relax\@svsec}{\interlinepenalty \@M \uppercase{#8}\par}% \uppercase
      \fi
    \else
      \@hangfrom{\hskip #3\relax{\rm\@svsec}}{\interlinepenalty \@M {\raggedright#8\par}}%
    \fi
  \endgroup
   \csname #1mark\endcsname{#7}%
   \addcontentsline{toc}{#1}{\ifnum #2>\c@secnumdepth \else %ajw
   \protect\numberline{\csname the#1\endcsname}\fi #7}%ajw
 \else
  \def\@svsechd{#6\hskip #3{\rm\@svsec}\if@levelone \uppercase{#8}\else #8.\hskip1em\fi%
  \csname #1mark\endcsname{#7}%
  \addcontentsline{toc}{#1}{\ifnum #2>\c@secnumdepth \else %ajw
  \protect\numberline{\csname the#1\endcsname}\fi#7}%ajw
  }\fi
 \@xsect{#5}}
% from latex.tex
%   \@ssect{INDENT}{BEFORESKIP}{AFTERSKIP}{STYLE}{ARG}
\def\@ssect#1#2#3#4#5{%
  \@tempskipa #3\relax
    \ifdim \@tempskipa>\z@
      \begingroup #4\@hangfrom{\hskip #1}{%
      \interlinepenalty \@M
      \if@levelone
        \uppercase{#5}%
      \else \uppercase{#5}%
      \fi\par}%
     \endgroup
    \else
      \def\@svsechd{#4\hskip #1\relax
      \if@levelone
        \uppercase{#5}%
      \else \uppercase{#5}%
      \fi}
    \fi
  \@xsect{#3}%
}
\newif\ifSFB@appendix
\def\appendix{\par
 \if|\@date|\else\vskip10pt{\@date\endgraf}\fi%
 \gdef\@date{}%
 \SFB@appendixtrue
 \@addtoreset{equation}{section}
 \@addtoreset{table}{section}
 \@addtoreset{figure}{section}
 \setcounter{secnumdepth}{0}
 \setcounter{section}{0}
 \setcounter{equation}{0}
 \setcounter{figure}{0}
 \setcounter{table}{0}
 \def\thesection{\Alph{section}}
 \renewcommand\thesubsection{\thesection\arabic{subsection}}
 \renewcommand\theequation{A.\thesection\arabic{equation}}
%%% marie get rid of these as quick fix for the endfloat problem
% \renewcommand\thefigure{A.\thesection\@arabic\c@figure}
% \renewcommand\thetable{A.\thesection\@arabic\c@table}
}
\newcounter{discounter}
\newenvironment{discussion}{\par\addvspace{30pt plus2pt minus1pt}%
\gdef\@email{}%
%%%%
 \refstepcounter{discounter}
 \setcounter{section}{0}
 \setcounter{equation}{0}
 \setcounter{figure}{0}
 \setcounter{table}{0}
%%%%
}{\Verticalskip\z@\par\addvspace{12pt plus2pt minus1pt}}
%
\newdimen\@topalign\@topalign11pt
\def\topalign#1{\global\advance\@topalign#1}
\def\makediscussionhead{\let\break\endgraf%
\TwoColumn[\hrule width\textwidth height.5pt\vskip12pt%
\ifnum\c@discounter=1\vskip18pt\noindent{\LARGE\bf Discussions}\endgraf\vskip18pt\fi
  {\reset@font\large\bf
   \begin{tabular}[t]{@{\hskip24pt}p{\columnwidth}@{}}\let\\=\author@nextline\@author 
   \\ \vspace*{-9pt} \@email\endgraf
   \end{tabular}
   \endgraf}
   \vskip 22pt
]\Verticalskip\@topalign
}
%
\def\@outputpage{%
\begingroup           % the \endgroup is put in by \aftergroup
  \let \protect \noexpand
  \@resetactivechars
  \global\let\@@if@newlist\if@newlist
  \global\@newlistfalse
  \@parboxrestore
  \shipout \vbox{%
    \set@typeset@protect
    \aftergroup \endgroup
    \aftergroup \set@typeset@protect
                                % correct? or just restore by ending
                                % the group?
  \if@specialpage
    \global\@specialpagefalse\@nameuse{ps@\@specialstyle}%
  \fi
  \if@twoside
    \ifodd\count\z@ \let\@thehead\@oddhead \let\@thefoot\@oddfoot
         \let\@themargin\oddsidemargin
    \else \let\@thehead\@evenhead
       \let\@thefoot\@evenfoot \let\@themargin\evensidemargin
    \fi
  \fi
  \reset@font
  \normalsize
  \normalsfcodes
  \let\label\@gobble
  \let\index\@gobble
  \let\glossary\@gobble
  \baselineskip\z@skip \lineskip\z@skip \lineskiplimit\z@
    \@begindvi
    \vskip \topmargin
    \moveright\@themargin \vbox {%
      \setbox\@tempboxa \vbox to\headheight{%
        \vfil
        \color@hbox
          \normalcolor
          \hb@xt@\textwidth{\@thehead}%
        \color@endbox
        }%                        %% 22 Feb 87
      \dp\@tempboxa \z@
      \box\@tempboxa
      \vskip \headsep
      \box\@outputbox
      \baselineskip \footskip
      \color@hbox
        \normalcolor
        \hb@xt@\textwidth{\@thefoot}%
      \color@endbox
      }%
    }%
  \global\let\if@newlist\@@if@newlist
  \global \@colht \textheight
  \stepcounter{page}%
  \let\firstmark\botmark
}
%
\newdimen\Verticalskip\Verticalskip\z@
\def\@outputdblcol{%
  \if@firstcolumn
    \global \@firstcolumnfalse
    \global \setbox\@leftcolumn \box\@outputbox
    \ifswitchcols\global\switchcolsfalse
       \ifdim\@partialpageht>0pt\advance\@partialpageht\Verticalskip\fi
       \vglue\@partialpageht
    \fi
  \else
    \global \@firstcolumntrue
    \setbox\@outputbox \vbox {%
                         \hbox to\textwidth {%
                           \hbox to\columnwidth {%
                             \box\@leftcolumn \hss}%
                           \hfil
                           \vrule \@width\columnseprule
                           \hfil
                           \hbox to\columnwidth {%
                             \box\@outputbox \hss}%
                                             }%
                              }%
    \@combinedblfloats
    \@outputpage
    \begingroup
      \@dblfloatplacement
      \@startdblcolumn
      \@whilesw\if@fcolmade \fi
        {\@outputpage
         \@startdblcolumn}%
    \endgroup
  \fi
}
%
\let\save@clearpage\clearpage
\let\save@outputpage\@outputpage
\newdimen\@partialpageht\@partialpageht10pc
\newif\ifswitchcols
\newbox\store@outputbox
\newdimen\storedboxheight
%
\def\store@outputpage{%
  \global\setbox\store@outputbox \vbox to \storedboxheight{%
      \leftline{\box\@outputbox}\vss}%
   \noindent%\fbox
   {\box\store@outputbox}%
   \par%\vrule width\textwidth height1pt
   \global\let\@outputpage\save@outputpage}
%
\def\balancepage{\@ifnextchar[{\balancecolpagebreak}{\balancecolpagebreak[0pt]}}
\def\balancecolpagebreak[#1]{\ifvmode\vadjust\fi{\vskip-\baselineskip
                             \leftline{\vrule width\typewidth height\draftrule depth0pt}%
                             \expandafter\ifdim#1=0pt\vfill\else\vskip#1\fi}\pagebreak}
%
\def\balanceandcontinue{%
   \par
   \ifdim\pagegoal=\maxdimen
   \else
      \global\let\@outputpage\store@outputpage
      \ifdim\@colht=\pagegoal
         \global\storedboxheight\pagetotal
      \else
         \global\storedboxheight\@colht
         \global\advance\storedboxheight-\pagegoal
         \global\advance\storedboxheight\pagetotal
      \fi
   \fi
   \onecolumn
   \global\switchcolstrue}
%
\def\TwoColumn{\@ifnextchar[{\@TwoColumn}{\@TwoColumn[]}}
\def\@TwoColumn[#1]{%
   \balanceandcontinue
   \let\clearpage\relax
   \def\@tempa{#1}%
   \ifx\@tempa\@empty
      \twocolumn
   \else
      #1\global\@partialpageht\pagetotal\twocolumn
   \fi
   \let\clearpage\save@clearpage
}
%  ****************************************
%  *               SKIPS                  *
%  ****************************************
%
\newskip\@indentskip
\newskip\smallindent
\newskip\@footindent
\newskip\@leftskip
\@indentskip=18pt
\smallindent=1.5em
\@footindent=\smallindent
\@leftskip=\z@

%  ****************************************
%  *                LISTS                 *
%  ****************************************
%
\setlength\leftmargini{   \z@ }
\setlength\leftmarginii{  1em }
\setlength\leftmarginiii{ 1em }
\setlength\leftmarginiv{  1em }
\setlength\leftmarginv{  1pc }
\setlength\leftmarginvi{  1pc }
\setlength\leftmargin{\leftmargini}
\setlength\labelsep{ 0.5em}
\labelwidth\leftmargini\advance\labelwidth-\labelsep
%
% label macros for Range-Left and Range-Right labels
\def\makeRLlabel#1{\rlap{#1}\hss}
\def\makeRRlabel#1{\hss\llap{#1}}
\def\makenewlabel#1{#1}
%
\def\@listI{\leftmargin\leftmargini
 \parsep \z@
 \topsep 6pt plus 1pt minus 1pt
 \itemsep \z@ plus .1pt
}
\let\@listi\@listI
\@listi
\def\@listii{\leftmargin\leftmarginii
 \topsep 6pt plus 1pt minus 1pt
 \parsep \z@
 \itemsep \z@ plus .1pt
}
\def\@listiii{\leftmargin\leftmarginiii
 \topsep 6pt plus 1pt minus 1pt
 \parsep \z@
 \partopsep \z@
 \itemsep \topsep
}
\def\@listiv{\leftmargin\leftmarginiv
}
\def\@listv{\leftmargin\leftmarginv
 \labelwidth\leftmarginv\advance\labelwidth-\labelsep
}
\def\@listvi{\leftmargin\leftmarginvi
 \labelwidth\leftmarginvi\advance\labelwidth-\labelsep
}
%
% ITEMIZE
\def\itemize{%
\topsep6pt%\vspace*{6pt}%\abovelistskipi6\p@ plus2pt%
%%%\belowlistskipi6\p@ plus2pt%
  \ifnum \@itemdepth >3 \@toodeep
  \else \advance\@itemdepth \@ne
   \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
   \list{\csname\@itemitem\endcsname}%
   {\let\makelabel\makeRRlabel}%
%   \parindent 1em%
%   \leftmargin 3.5em\labelsep 0.5em\itemindent 2.0em\labelwidth 0.5em%
   \labelwidth23pt	
   \labelsep5pt
   \leftmargin\z@
   \advance\leftmargin\labelwidth
   \advance\leftmargin\labelsep
   \leftskip\leftmargin
%\bottomsep6pt
  \fi}
\def\labelitemi{$\bullet$}
\def\labelitemii{--}
\def\labelitemiii{$\cdot$}
\def\labelitemiv{*}
%
% ENUMERATE
% SFB -- enumerate with optional argument to set left margin...
\def\enumerate{\ifnum \@enumdepth >3 \@toodeep \else
  \advance\@enumdepth \@ne
  \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
 \fi
 \@ifnextchar [{\@enumeratetwo}{\@enumerateone}%
}
\def\@enumeratetwo[#1]{%
 \list{\csname label\@enumctr\endcsname}%
  {\settowidth\labelwidth{[#1]}
   \leftmargin18pt%\z@
   \advance\leftmargin\labelwidth
   \advance\leftmargin\labelsep
   \listparindent 1em
   \usecounter{\@enumctr}
   \let\makelabel\makenewlabel}
}
\def\@enumerateone{%
 \list{\csname label\@enumctr\endcsname}%
  {\usecounter{\@enumctr}
  \settowidth\labelwidth{(9)}
   \leftmargin18pt%\z@
   \advance\leftmargin\labelwidth
   \advance\leftmargin\labelsep
   \listparindent 1em
   \let\makelabel\makenewlabel}}
%
\def\labelenumi{\theenumi}
\def\theenumi{(\arabic{enumi})}% \roman
\def\labelenumii{\theenumii}
\def\theenumii{(\alph{enumii})}
\def\p@enumii{\theenumi}
\def\labelenumiii{\theenumiii}
\def\theenumiii{(\arabic{enumiii})}
\def\p@enumiii{\theenumi(\theenumii)}
\def\labelenumiv{\theenumiv}
\def\theenumiv{(\Alph{enumiv})}
\def\p@enumiv{\p@enumiii\theenumiii}
%
% DESCRIPTION
\def\descriptionlabel#1{\reset@font\bf #1}
\def\description{\list{}{\labelwidth\z@
  \leftmargin 1em
  \itemindent -1em
}}
\let\enddescription\endlist

%  ****************************************
%  *            DISPLAYED TEXT            *
%  ****************************************
%
% VERSE
\def\verse{\let\\=\@centercr
 \list{}{\itemsep\z@
  \itemindent -\@indentskip
  \listparindent \itemindent
  \rightmargin\leftmargin
  \advance\leftmargin \@indentskip}\item[]}
\let\endverse\endlist
%
% QUOTATION
\def\quotation{\list{}{\listparindent \smallindent
%  \itemindent\listparindent
  \leftmargin\z@\rightmargin\leftmargin
  \parsep 0pt plus 1pt}\item[]\reset@font\small}
\let\endquotation=\endlist
%
% QUOTE
\def\quote{\list{}{\leftmargin\z@\rightmargin\leftmargin}\item[]%
  \reset@font\small}
\let\endquote=\endlist
%
% BSL QUOTE
\newenvironment{lquote}{\begin{list}{}%
  {\setlength{\leftmargin}{2em}}
  \item[]}
  {\end{list}}
%
%  ****************************************
%  *         OTHER ENVIRONMENTS           *
%  ****************************************
%
% THEOREM
\def\@begintheorem#1#2{\reset@font\rm \trivlist \item[\hskip \labelsep
  {\reset@font\bf #1\ #2.}]}
\def\@opargbegintheorem#1#2#3{\reset@font\rm \trivlist
  \item[\hskip \parindent{\reset@font~\bf #1\ #2.\ (#3)}]}% \labelsep
\def\@endtheorem{\endtrivlist}
%
% PROOF
\def\proof{\reset@font\rm \trivlist \item[\hskip \parindent%labelsep
  {\reset@font~~\it Proof.}]}
\def\endproof{\hspace*{1em}%{\begin{picture}(6.5,6.5)%
%  \put(0,0){\framebox(6.5,6.5){}}\end{picture}}
\endtrivlist}
\@namedef{proof*}{\reset@font\rm \trivlist \item[\hskip \labelsep
  {\reset@font\it Proof.}]}
\@namedef{endproof*}{\endtrivlist}
\def\proofbox{\begin{picture}(6.5,6.5)%
  \put(0,0){\framebox(6.5,6.5){}}\end{picture}}
%
% TITLEPAGE
\def\titlepage{\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
  \else \newpage \fi \thispagestyle{empty}\c@page\z@}
\def\endtitlepage{\if@restonecol\twocolumn \else \newpage \fi}

% ARRAY AND TABULAR
%
\arraycolsep   5pt
\tabcolsep     6pt
\arrayrulewidth .5pt
\doublerulesep 1.5pt

\def\tabular{\def\@halignto{}
 \def\Hline{\noalign{\ifnum0=`}\fi
  \vskip 3pt
  \hrule \@height \arrayrulewidth
  \vskip 1.25pt
  \hrule \@height \arrayrulewidth
  \vskip 3pt
  \futurelet \@tempa\@xhline}
 \def\hline{\noalign{\ifnum0=`}\fi
  \vskip 1pt
  \hrule \@height \arrayrulewidth
  \vskip 3pt
  \futurelet \@tempa\@xhline}
 \def\fullhline{\noalign{\ifnum0=`}\fi
  \vskip 3pt
  \hrule \@height \arrayrulewidth
  \vskip 3pt
  \futurelet \@tempa\@xhline}
 \def\@xhline{\ifx\@tempa\hline
   \vskip -6pt
   \vskip \doublerulesep
  \fi
  \ifnum0=`{\fi}}
  \def\@arrayrule{\@addtopreamble{\hskip -.5\arrayrulewidth
%                                  \vrule \@width \arrayrulewidth
                                  \hskip .5\arrayrulewidth}}
\centering\@tabular
}
\def\endtabular{}%\egroup\egroup \egroup\par\vspace*{-20pt}}
%\expandafter \let \csname endtabular*\endcsname = \endtabular
% TABBING
%
\@namedef{tabular*}#1{\figcaptionfont%
 \def\Hline{\noalign{\ifnum0=`}\fi
  \vskip 3pt
  \hrule \@height 2pt%\arrayrulewidth
%%   \vskip 1.25pt
%%   \hrule \@height \arrayrulewidth
   \vskip 3pt
  \futurelet \@tempa\@xhline}
 \def\hline{\noalign{\ifnum0=`}\fi
  \vskip 1pt
  \hrule \@height .5pt%\arrayrulewidth
  \vskip 3pt
  \futurelet \@tempa\@xhline}
 \def\fullhline{\noalign{\ifnum0=`}\fi
  \vskip 3pt
  \hrule \@height \arrayrulewidth
  \vskip 3pt
  \futurelet \@tempa\@xhline}
 \def\@xhline{\ifx\@tempa\hline
   \vskip -6pt
   \vskip \doublerulesep
  \fi
  \ifnum0=`{\fi}}
  \def\@arrayrule{\@addtopreamble{\hskip -.5\arrayrulewidth
%                                  \vrule \@width \arrayrulewidth
                                  \hskip .5\arrayrulewidth}}
\centering
 \setlength\dimen@{#1}%
   \edef\@halignto{to\the\dimen@}\@tabular}
%
\tabbingsep \labelsep

% MINIPAGE
%
\skip\@mpfootins = \skip\footins

% FRAMEBOX
%
% Space left between box and text by \fbox and \framebox.
\fboxsep = 3pt
% Width of rules in box made by \fbox and \framebox.
\fboxrule = \arrayrulewidth

%  ****************************************
%  *         TITLE AND ABSTRACT           *
%  ****************************************
%%\newif\ifthanks%\thankstrue%%%%false
%
\global\newcount\affilcount
\global\newcount\tempcount
\affilcount=0
\def\affiliation#1{\global\advance\affilcount by \@ne%
%\leavevmode$^{\the\affilcount}$
 \expandafter\gdef\csname affil\romannumeral\the\affilcount\endcsname{#1}}%
%
\global\newcount\xaffilcount
\def\@printaffil{%
   \global\xaffilcount\affilcount
    \ifnum\affilcount>0%
       \tempcount=1%
       \loop%
       \ifnum\affilcount>0%
          \footnotetext{$^{\the\tempcount}$\csname affil\romannumeral\the\tempcount\endcsname}%
%%%             \vskip0pt
          \global\advance\affilcount by -1%
          \advance\tempcount by 1 %
       \repeat%
    \fi}%
%%%%%%%%%%%%%%
\def\maketitle{\par%
 \begingroup
\renewcommand{\thefootnote}{\fnsymbol{footnote}}
  \if@twocolumn
   \twocolumn[\vspace*{17pt}\@maketitle]
  \else
   \newpage
   \global\@topnum\z@
   \@maketitle
  \fi
  \thispagestyle{titlepage}
\renewcommand{\thefootnote}{\fnsymbol{footnote}}
\setcounter{footnote}{0}
\@printaffil\endgraf
 \@thanks%
\endgroup%
\renewcommand{\thefootnote}{\arabic{footnote}}
\setcounter{footnote}{\xaffilcount}
 \ifSFB@referee \clearpage\setcounter{page}{1}\fi
 \let\maketitle\relax
 \let\@maketitle\relax
 \gdef\@author{}
 \gdef\@title{}
 \let\thanks\relax}
%
%
\def\@maketitle{\newpage
 \vspace*{25.7pt}
 {\raggedright \sloppy
  {\reset@font\huge\raggedright\bfseries\@title \par}
  \vspace*{28.7pt}
  {{\authorfont\raggedright\@author\par}
   \par}
  \vspace*{35pt}
 }%
 \unvbox\absbox
 \vspace*{10pt}
 \unvbox\keywdbox
 \par\noindent
 \vskip2.2pc
}%

\def\noaffil{yes}
\def\YES{yes}
\def\author@nextline{          %DE's definition
\ifx\noaffil\YES
\tabularnewline[4pt]
\gdef\noaffil{no}\else
\tabularnewline[0pt]
\fi\reset@font\ifbackmatter\it\fi\ignorespaces\mathversion{normal}%
\let\\=\author@nextline}       %let command added by NT to bring
%%%%%%%%%%%%%
\def\thanks#1{\footnotemark%
    \protected@xdef\@thanks{\@thanks%
        \protect\footnotetext[\the\c@footnote]{#1}}%
}
\def\@makefnmark{\hbox{\@textsuperscript{\normalfont\@thefnmark}}}
\newbox\absbox
\def\abstract{\global\setbox\absbox\vbox\bgroup\hsize\textwidth%
\leftskip10pc\noindent\rule{30pc}{1pt}\vspace*{7.2pt}\par
\abstractfont\leftskip10pc\noindent\ignorespaces}
\def\endabstract{\par\vskip2pt\noindent\rule{30pc}{1pt}\par\egroup}
%
\newbox\keywdbox
\def\keywords{\global\setbox\keywdbox\vbox\bgroup\hsize\textwidth\keywordfont\leftskip10pc\noindent
{\bfseries KEY WORDS:}\hskip1em\ignorespaces}
\def\endkeywords{\egroup}
%
\def\nokeywords{\ifSFB@keywords\else	% 
 \if@twocolumn \start@SFBbox\addvspace{37pt}\finish@SFBbox \fi
 \@thanks
 \gdef\@thanks{}\fi
}
\def\start@SFBbox{\@next\@currbox\@freelist{}{}%
 \global\setbox\@currbox
 \vbox\bgroup
  \hsize \textwidth
  \@parboxrestore
}
\def\finish@SFBbox{\par\vskip -\dbltextfloatsep
  \egroup
  \global\count\@currbox\tw@
  \global\@dbltopnum\@ne
  \global\@dbltoproom\maxdimen\@addtodblcol
  \global\vsize\@colht
  \global\@colroom\@colht
}

%  ****************************************
%  *            PAGE STYLES               *
%  ****************************************
%
\mark{{}{}}
\gdef\@author{\mbox{}}
\def\author{\@ifnextchar [{\@authortwo}{\@authorone}}
\def\@authortwo[#1]#2{\gdef\@author{#2}\gdef\@shortauthor{{#1}}}
\def\@authorone#1{\gdef\@author{#1}\gdef\@shortauthor{{#1}}}
\gdef\@shortauthor{}
\gdef\@title{\mbox{}}
\def\title{\@ifnextchar [{\@titletwo}{\@titleone}}
\def\@titletwo[#1]#2{\gdef\@title{{#2}}\gdef\@shorttitle{{#1}}}
\def\@titleone#1{\gdef\@title{{#1}}\gdef\@shorttitle{{#1}}}
\gdef\@shorttitle{}
\def\@email{}
\def\@emailx{}
\newcounter{emailcnt}
\newcounter{emailxcnt}
\gdef\emailbreak{}
\newcount\tempcount
\def\printemails{
\tempcount=0
\@whilenum\c@emailcnt>0
\do{%
\global\advance\tempcount 1
\\
\@emailsymbol{\the\tempcount}{\it email:}\  \csname @@email\romannumeral\the\tempcount\endcsname
\global\addtocounter{emailcnt}{-1}}
}
%%
\def\printemailxs{
\tempcount=0
\@whilenum\c@emailxcnt>0
\do{%
\global\advance\tempcount 1
\\
{\it email:}\  \csname @@emailx\romannumeral\the\tempcount\endcsname
\global\addtocounter{emailxcnt}{-1}}
}

\def\emailx#1{\global\addtocounter{emailxcnt}{1}%
	        \expandafter\gdef\csname @@emailx\romannumeral\theemailxcnt\endcsname{#1}%
}

\def\email#1{\global\addtocounter{emailcnt}{1}%
	        \expandafter\gdef\csname @@email\romannumeral\theemailcnt\endcsname{#1}%
}

\def\xemail#1{\refstepcounter{emailcnt}\ifx\@email\@empty%
			    \protected@xdef\@email{\ifbackmatter{\rm email:}\else\@emailsymbol{\c@emailcnt}{\it email:}\fi\ #1 }%
			\else%	
			    \protected@xdef\@email{\@email\ifbackmatter{\rm
				email:}\else\@emailsymbol{\c@emailcnt}{\it email:}\fi\ #1 }% 
			\fi}%
%
\def\@jnum{}
\def\jnum#1{\gdef\@jnum{#1}}
%
\def\@doi{000}
\def\doi#1{\gdef\@doi{#1}}
\def\@date{}
\def\date#1{\gdef\@date{#1}}
\def\jvol#1{\gdef\@jvol{#1}}
\jvol{}
\def\issn#1{\gdef\@issn{#1}}
\issn{0272-4332}
\def\issnyear#1{\gdef\@issnyear{#1}}
\issnyear{}
%\gdef\@volume{000}
\def\artmonth#1{\gdef\@month{#1}}
\gdef\@month{000}
\def\editor#1{\gdef\@editor{#1}}
\gdef\@editor{XxX}
\AtEndDocument{\if|\@date|\else\vskip10pt{\centering\@date\endgraf}\fi}
\def\tod@y{\today}
\def\microfiche#1{\gdef\@microfiche{#1}}
\gdef\@microfiche{}
\def\pubyear#1{\gdef\@pubyear{#1}}
\gdef\@pubyear{0000}
\def\pagerange#1{\gdef\@pagerange{#1}}
\gdef\@pagerange{000--000}
\def\journal#1{\gdef\@journal{#1}}
\def\plate#1{\gdef\pl@te{#1}}
\ifx\CUP@mtlplain@loaded\undefined
  \def\@printed{\qquad Printed \tod@y\qquad
    (MN \LaTeX\ style file v\@version)}
\else
  \def\@printed{}
\fi
\def\jname#1{\gdef\@jname{#1}}
\jname{}
%%%%%%%%%%%%
\def\Copyright{{$\raise1.15pt\hbox{\fontsize{5.55}{6}\selectfont\ooalign{\hfil\hskip-1pt\raise-0.05ex\hbox{\fontsize{4.05}{6}\selectfont C}%
                \hfil\crcr \mathhexbox20D}}$}}

\def\copyright#1{\gdef\@copyright{#1}}
\copyright{}
%%%%%%%%%%%
\def\@ddell{}
\if@letters
  \def\@ddell{L}
\fi
\def\ps@headings{\let\@mkboth\markboth
 \def\@oddhead{\rhfont{\bfseries\@shorttitle}\hfill{\foliofont\thepage}}
 \def\@oddfoot{}%
 \def\@evenhead{\rhfont\@ddell\hbox to
 0pt{\foliofont\thepage}\hspace{1.5em}\hfill{\@shortauthor}}
 \def\@evenfoot{}%
 \def\sectionmark##1{\markboth{##1}{}}
 \def\subsectionmark##1{\markright{##1}}}
%
\def\ps@myheadings{\let\@mkboth\@gobbletwo
 \def\@oddhead{\Large\hfill\it\rightmark\hspace{1.5em}\rm\@ddell\thepage}
 \def\@oddfoot{\small\copyright\ \@pubyear\ RAS, MNRAS {\bf \@volume}, \@pagerange\hfil}
 \def\@evenhead{\Large\@ddell\thepage\hspace{1.5em}\it\leftmark\hfill}
 \def\@evenfoot{\hfil\small\copyright\ \@pubyear\ RAS, MNRAS {\bf \@volume}, \@pagerange}
 \def\sectionmark##1{}
 \def\subsectionmark##1{}}
%%
\font\mycirc =  lcircle10 at 6pt% cioptic%
\def\Copyright{\leavevmode\raise3pt\hbox{\reset@font\mycirc\char'151}\llap{\raise1.25pt\hbox
to 7.75pt{\tiny C}}}
%%%
\def\pagenum{\ifnum\c@page < 10 000\else\ifnum\c@page < 100 00\else\ifnum\c@page < 1000 0\fi\fi\fi\thepage}
%%
\def\ps@titlepage{\let\@mkboth\@gobbletwo%
      \def\@evenfoot{\hphantom{}%
                     \hfill{\dropfoliofont}\hfill%
                     }%
      \def\@oddfoot{\hphantom{{\short 0272-4332/00/0100-\pagenum\$22.00/1\ {\Copyright}\ \@copyright}}%
                    \hfill{\dropfoliofont}\hfill%
                   }%
      \def\@evenhead{\lower4.7pt\hbox{\creditlinefont \@jname, Vol. \@jvol, No. \@jnum, \@pubyear}\hfill%
                     \lower4.7pt\hbox{\creditlinefont{\rm DOI:\ \@doi}}}%
      \def\@oddhead{\lower4.7pt\hbox{\creditlinefont \@jname, Vol. \@jvol, No. \@jnum, \@pubyear}\hfill%
                    \lower4.7pt\hbox{\creditlinefont{\rm DOI:\ \@doi}}}}
%
\def\ps@plate{\let\@mkboth\@gobbletwo
  \def\@oddhead{\footnotesize\pl@te\hfill}
  \def\@oddfoot{\small\copyright\ \@pubyear\ RAS, MNRAS {\bf \@volume},
    \@pagerange\hfil}
  \def\@evenhead{\footnotesize\pl@te\hfill}
  \def\@evenfoot{\hfil\small\copyright\ \@pubyear\ RAS, MNRAS {\bf \@volume},
    \@pagerange}
}

%  ****************************************
%  *         TABLE OF CONTENTS, ETC.      *
%  ****************************************
%
\def\@pnumwidth{1.55em}
\def\@tocrmarg {2.55em}
\def\@dotsep{4.5}
\setcounter{tocdepth}{2}
\def\@undottedtocline#1#2#3#4#5{\ifnum #1>\c@tocdepth
 \else
  \vskip \z@ plus .2pt
  {\hangindent #2\relax
   \rightskip \@tocrmarg \parfillskip -\rightskip
   \parindent #2\relax \@afterindenttrue
   \interlinepenalty\@M \leavevmode
   \@tempdima #3\relax #4\nobreak \hfill \nobreak
   \hbox to\@pnumwidth{\hfil\reset@font\rm \ }\par}\fi}
\def\tableofcontents{\@restonecolfalse
 \if@twocolumn\@restonecoltrue\onecolumn\fi
 \section*{CONTENTS} \@starttoc{toc}
 \if@restonecol\twocolumn\fi \par\vspace{12pt}}
\def\l@part#1#2{\addpenalty{-\@highpenalty}
 \addvspace{2.25em plus 1pt}
 \begingroup
  \parindent \z@ \rightskip \@pnumwidth
  \parfillskip -\@pnumwidth
  {\reset@font\normalsize\rm
   \leavevmode \hspace*{3pc}
   #1\hfil \hbox to\@pnumwidth{\hss \ }}\par
   \nobreak \global\@nobreaktrue
   \everypar{\global\@nobreakfalse\everypar{}}\endgroup}
\def\l@section#1#2{\addpenalty{\@secpenalty}
 \@tempdima 1.5em
 \begingroup
  \parindent \z@ \rightskip \@pnumwidth
  \parfillskip -\@pnumwidth \reset@font\rm \leavevmode
  \advance\leftskip\@tempdima \hskip -\leftskip
  #1\nobreak\hfil \nobreak\hbox to\@pnumwidth{\hss \ }\par
 \endgroup}
\def\l@subsection{\@undottedtocline{2}{1.5em}{2.3em}}
\def\l@subsubsection{\@undottedtocline{3}{3.8em}{3.2em}}
\def\l@paragraph{\@undottedtocline{4}{7.0em}{4.1em}}
\def\l@subparagraph{\@undottedtocline{5}{10em}{5em}}
\def\listoffigures{\@restonecolfalse
 \if@twocolumn\@restonecoltrue\onecolumn\fi
 \section*{LIST OF FIGURES\@mkboth{LIST OF FIGURES}{LIST OF FIGURES}}
 \@starttoc{lof} \if@restonecol\twocolumn\fi}
\def\l@figure{\@undottedtocline{1}{1.5em}{2.3em}}
\def\listoftables{\@restonecolfalse
 \if@twocolumn\@restonecoltrue\onecolumn\fi
 \section*{LIST OF TABLES\@mkboth{LIST OF TABLES}{LIST OF TABLES}}
 \@starttoc{lot} \if@restonecol\twocolumn\fi}
\let\l@table\l@figure

%  ****************************************
%  *             BIBLIOGRAPHY             *
%  ****************************************
%
%										
\def\eqnbreaktop#1{\break\vrule width\columnwidth height.5pt\par%
\vrule height#1 width0pt\par
\noindent\vrule width\columnwidth height.5pt\par\vskip6pt\noindent
\ignorespaces}
\def\aurepeat{\leavevmode\raise3pt\hbox{\vrule height.25pt width2.25pc}}
\newcounter{dummy}
\def\thebibliography#1{\section*{\refname}%
\bibliofont
 \addcontentsline{toc}{section}{References}
 \list{}{\usecounter{dummy}%
         \labelwidth\z@
         \leftmargin 20pt
         \itemsep \z@
         \itemindent-\leftmargin}
 \reset@font
 \parindent\z@
 \parskip\z@\relax%  plus .1pt
%%% \def\newblock{\hskip .11em plus .33em minus .07em}
 \sloppy\clubpenalty4000\widowpenalty4000
 \sfcode`\.=1000\relax
}
%% \let\endthebibliography=\endlist
%% \def\@biblabel#1{\hspace*{\labelsep}[#1]}

% new column specifiers to be used with dcolumn
\if@usedcolumn%
 \usepackage{dcolumn}
 \newcolumntype{d}[1]{D{.}{.}{#1}}
 \newcolumntype{.}{D{.}{.}{-1}}
 \newcolumntype{,}{D{,}{,}{2}}
\fi

%  ****************************************
%  *              THE INDEX               *
%  ****************************************
%
\newif\if@restonecol
\def\theindex{\section*{INDEX}
 \addcontentsline{toc}{section}{INDEX}
 \reset@font\footnotesize \parindent\z@ \parskip\z@ plus .1pt\relax
 \let\item\@idxitem}
\def\@idxitem{\par\hangindent 1em}
\def\subitem {\par\hangindent 1em \hspace*{1em}}
\def\subsubitem{\par\hangindent 1em\hspace*{1em}}
\def\endtheindex{\if@restonecol\onecolumn\else\clearpage\fi}
\def\indexspace{\par\vskip\baselineskip \relax}

%  ****************************************
%  *             FOOTNOTES                *
%  ****************************************
%% \def\@fnsymbol#1{\ifcase#1\or *\or \dagger\or \ddagger\or
%%    \SS\or \|\or \#\or **\or \dagger\dagger\or \ddagger\ddagger 
%%    \or \SS\SS \or \|\hskip-1pt\|\or \#\#\or ***\or 
%%    \dagger\dagger\dagger\or \ddagger\ddagger\ddagger\else\@ctrerr\fi\relax}
%
\def\footnoterule{}%\kern-3\p@ \hrule width 3.5pc height .5\p@ \kern 5\p@}

\renewcommand{\thefootnote}{\mbox{{$\arabic{footnote}$}}}
%\renewcommand{\thempfootnote}{\mbox{{${\alph{mpfootnote}}$}}}

\def\@fnsymbol#1{\ifcase#1\or \mbox{${^{\star}}$}\or
   \dagger\or \ddagger\or
   \S \or \P \or \|\or \mbox{$^{\star\star}$}\or \dagger\dagger
   \or \ddagger\ddagger\or \S\S\or \P\P\or \|\|\else ***
   \fi\relax}
%
\def\@emailsymbol#1{\ifcase#1\or *\or **\or ***\or ****\or *****\or ******\or *******\or ********\or *********\or **********
   \fi\relax}

\providecommand\@makefntext[1]{}
\long\def\@makefntext#1{{\leavevmode\smash{$\m@th^{\@thefnmark}$}}#1\endgraf}
%%%%****************************************
%  *         FIGURES AND TABLES           *
%  ****************************************
%
\setcounter{topnumber}{5}
\def\topfraction{.9}
\setcounter{bottomnumber}{5}
\def\bottomfraction{.9}
\setcounter{totalnumber}{10}
\def\textfraction{.1}
\def\floatpagefraction{.9}
\setcounter{dbltopnumber}{2}
\def\dbltopfraction{.9}
\def\dblfloatpagefraction{.9}

\newif\ifst@rredfloat
\newcounter{table}
\renewcommand\thetable{\@Roman\c@table\ }
%%%\def\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{1}
\def\fstyle@table{\reset@font\fontsize{8}{10pt}\selectfont\rm}
\def\thead#1{\reset@font#1}
\def\ext@table{lot}
\def\fnum@table{Table \thetable.}
%%
\def\FigName{figure}
%
\long\def\@makecaption#1#2{%
    \ifx\FigName\@captype
%      \vskip\abovecaptionskip
      \SFB@makefigurecaption{#1}{#2}%
%      \vskip\belowcaptionskip
    \else
      \SFB@maketablecaption{#1}{#2}%
%      \vskip\belowcaptionskip
    \fi
}
%
\newenvironment{table}
               {\def\footnoterule{\kern-3\p@ \hrule width 3pc height \z@ \kern 5\p@}%  
                \figcaptionfont\centering\fstyle@table%
                \@float{table}}
               {\end@float}
%%
\newenvironment{table*}
               {\def\footnoterule{\kern-3\p@ \hrule width 3pc height \z@ \kern 5\p@}%
			   \centering\fstyle@table\figcaptionfont%
			   \@dblfloat{table}}
               {\end@dblfloat}

\newcounter{figure}
\def\thefigure{\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{2}
\def\fstyle@figure{\reset@font\small\rm}
\def\ext@figure{lof}
\def\fnum@figure{{\bf Fig. \thefigure}}
%%
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
%%
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
%%

\long\def\SFB@makefigurecaption#1#2{\figcaptionfont\vskip 6pt%
  \setbox\@tempboxa\hbox{\reset@font{\bf #1}%
    \if@cont \ -- {\it continued}\else{\bf.}\hskip1em\fi{\rm #2}\par}%
 \ifdim \wd\@tempboxa >\hsize%
   {\reset@font{\bf#1}%
     \if@cont \ -- {\it continued}\else{\bf.}\hskip1em\fi{\rm #2}\par}%
 \else%
   \ifst@rredfloat%
     \hbox to\hsize{\hfill\box\@tempboxa\hfill}%%
   \else%
     \hbox to\hsize{\hfill\box\@tempboxa\hfill}%
   \fi%
 \fi%
% \vskip 6pt%
 \@contfalse%
}%
%% %%%%%%%%%%%%%
\newbox\tempbox
\long\def\SFB@maketablecaption#1#2{\figcaptionfont\vskip 6pt
%    \ifdim\wd\hsize<27pc\leftskip12pc
{\reset@font\centering{\bf #1}\ %%\par\vskip1pt
    {#2}\par}
%\fi%
%%%%\ifdim\wd\tempbox<\hsize\leftskip10pc{\box\tempbox}\else\leftskip12pt{\box\tempbox}\fi
 \vskip 0pt
 \@contfalse
}
%%%
\newif\if@cont
\def\caption{\refstepcounter\@captype \@dblarg{\@caption\@captype}}
\def\contcaption{\@conttrue\SFB@caption\@captype}
%%%%
\long\def\SFB@caption#1#2{%\par
 \begingroup
  \@parboxrestore
  \reset@font\normalsize
  \@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #2}\par
 \endgroup}
%
%  ****************************************
%  *           CUP specials               *
%  ****************************************
%
%
%  The following code is taken from authordate.sty, by David Rhead, of
%  the Cripps Computing Centre (February 1990), with some modifications
%  by ggw. It is intended for documents
%  that use the author-date citation system. When used in conjunction with
%  authordate1.bst, ... , authordate4.bst, it allows citations of the
%  form \shortcite{bloggs-60} as well as the usual \cite{bloggs-60}.
%  When
%       ... \cite{bloggs-60} ... \shortcite{bloggs-60} ...
%  appears in the input file,
%       ... (Bloggs, 1960) ... (1960) ...
%  appears in the final document.
%
%  Code taken from apalike.sty.
%
%% \def\@cite#1#2{(#1\if@tempswa , #2\fi)}
%% \def\@biblabel#1{}

% Set length of hanging indentation for bibliography entries
%

%% \if@usenatbib\relax\else%
%% %% \newlength{\bibhang}
%% %% \setlength{\bibhang}{2em}
%% \fi

%  Define default title and page-head text.
%
\newcommand\refname{REFERENCES}

%  Code taken from aaai.sty.
%
% don't box citations, add space between multiple citations, separate with ;
\def\@citex[#1]#2{%
  \if@filesw\immediate\write\@auxout{\string\citation{#2}}\fi
  \def\@citea{}\@cite{\@for\@citeb:=#2\do
    {\@citea\def\@citea{; }\@ifundefined
       {b@\@citeb}{{\reset@font\bf ?}\@warning
       {Citation `\@citeb' on page \thepage \space undefined}}%
{\csname b@\@citeb\endcsname}}}{#1}}
% Allow short (name-less) citations, when used in
% conjunction with a bibliography style that creates labels like
%       \citename{<names>, }<year>
\let\@internalcite\cite
\def\cite{\def\citename##1{##1}\@internalcite}
\def\shortcite{\def\citename##1{}\@internalcite}

% RM for BSP note at end of article
\def\bsp{\vspace{0.5cm}\reset@font\normalsize\noindent This paper
has been typeset from a \TeX / \LaTeX\ file prepared by the author.}


%  ****************************************
%  *                FLEQN                 *
%  ****************************************
%
% FLEQN DOCUMENT-STYLE OPTION - released 9 March 1987
%    for LaTeX version 2.09
% Copyright (C) 1987 by Leslie Lamport
%
% The following commands make the displayed math environments flush left,
% with an indentation of \mathindent from the prevailing left margin.

\def\[{\relax\ifmmode\@badmath\else\begin{trivlist}\item[]\leavevmode
  \hbox to\linewidth\bgroup\hfil$
  \displaystyle
  \hskip\mathindent\bgroup\fi}

\def\]{\relax\ifmmode \egroup $\hfil
       \egroup \end{trivlist}\else \@badmath \fi}

\def\equation{\refstepcounter{equation}\trivlist \item[]\leavevmode
  \hbox to\linewidth\bgroup\hfil $
  \displaystyle
\hskip\mathindent}


\def\endequation{$\hfil
           \displaywidth\linewidth\@eqnnum\egroup \endtrivlist}

\def\longequation{\refstepcounter{equation}\trivlist \item[]\leavevmode
  \hbox to\textwidth\bgroup\hfil $
  \displaystyle
\hskip\mathindent}


\def\endlongequation{$\hfil
           \displaywidth\linewidth\@eqnnum\egroup \endtrivlist}

\def\eqnarray{\stepcounter{equation}\let\@currentlabel=\theequation
\global\@eqnswtrue
\global\@eqcnt\z@\tabskip\mathindent\let\\=\@eqncr
\abovedisplayskip\topsep\ifvmode\advance\abovedisplayskip\partopsep\fi
\belowdisplayskip\abovedisplayskip
\belowdisplayshortskip\abovedisplayskip
\abovedisplayshortskip\abovedisplayskip
$$\halign
to \linewidth\bgroup\@eqnsel\hskip\@centering$\displaystyle\tabskip\z@
  {##}$&\global\@eqcnt\@ne \hskip 2\arraycolsep \hfil${##}$\hfil
  &\global\@eqcnt\tw@ \hskip 2\arraycolsep $\displaystyle{##}$\hfil
   \tabskip\@centering&\llap{##}\tabskip\z@\cr}

\def\endeqnarray{\@@eqncr\egroup
 \global\advance\c@equation\m@ne$$\global\@ignoretrue}

\newdimen\mathindent
\mathindent = \z@

%  ****************************************
%  *            MISCELLANEOUS             *
%  ****************************************
%
% DATE
\def\today{\number\day\ \ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December
 \fi \ \number\year}

% REVISED AUTHOR QUERY COMMAND
% Queries in text are printed in large bold font, in square brackets
% Queries in references are printed as bold number followed by text
 \DeclareFixedFont{\largerm}{OT1}{cmss}{m}{n}{17}
 \DeclareFixedFont{\largeital}{OT1}{cmss}{m}{sl}{17}
 \newcommand{\aquery}[1]{\mbox{\largerm {[#1]}}}
 \newcommand{\BSLquery}[1]{\aquery{#1}}
 %For backwards compatibility with old query command
 \newcommand{\tquery}[1]{\mbox{\largeital{[#1]}}}
 %Typesetter query: same format
 \newenvironment{query}{\vspace{5mm} \noindent {\bf Queries}\\}{}
 \newcommand{\authorquery}[2]{\noindent{{\bf #1} #2}\\}

% definition required for newarray.sty
\gdef\d@llar{}

%  ****************************************
%  *           INITIALIZATION             *
%  ****************************************
%
%
\newif\ifbackmatter
\def\backmatter{%
	\global\backmattertrue
	}
%
%
%%%NHT: this code needs fixed.
%%%     At the moment the zif is always undefined, so it does nothing!
\newif\ifCUPmtlplainloaded
\ifx\CUP@mtlplain@loaded\undefined
\else
  \global\CUPmtlplainloadedtrue
  \input mnsym
\fi
%
\flushbottom
\ps@headings
\ifSFB@galley
 \raggedbottom
 \ps@empty
\fi
%\ifSFB@referee
% \raggedbottom
%\fi
\pagenumbering{arabic}
\if@twocolumn
 \twocolumn
 \sloppy
\else
 \onecolumn
\fi
\frenchspacing
%%%%
\usepackage{theorem}
\newcounter{theorem}
\renewcommand\thetheorem{\arbic{theorem}}
\newtheorem{theorem}{Theorem}%[theorem]
\newtheorem{algorithm}{Algorithm}%[theorem]
\newtheorem{corollary}{Corollary}%[section]%[theorem]
\newtheorem{fact}{Fact}%[section]
\newtheorem{property}{Property}%[section]
\newtheorem{result}{Result}%[section]
\newtheorem{lemma}{Lemma}%[section]
%\newtheorem{proof}{Proof}
%%%
\newtheorem{assumption}{Assumption}%[section]
\newtheorem{case}{Case}%[section]
\newtheorem{axiom}{Axiom}%[section]
\newtheorem{change}{Change}%[section]
\newtheorem{claim}{Claim}%[section]
\newtheorem{conjecture}{Conjecture}%[section]
\newtheorem{definition}{Definition}%[section]
\newtheorem{example}{Example}%[section]
\newtheorem{formula}{Formula}%[section]
\newtheorem{hypothesis}{Hypothesis}%[section]
\newtheorem{method}{Method}%[section]
\newtheorem{model}{Model}%[section]
\newtheorem{notation}{Notation}%[section]
\newtheorem{note}{Note}%[section]
\newtheorem{objective}{Objective}%[section]
\newtheorem{observation}{Observation}%[section]
\newtheorem{principle}{Principle}%[section]
\newtheorem{problem}{Problem}%[section]
\newtheorem{procedure}{Procedure}%[section]
\newtheorem{remark}{Remark}%[section]
\newtheorem{step}{Step}%[section]

\gdef\PROOF{Proof}
\gdef\ALGO{Algorithm}
\gdef\CORO{Corollary}
\gdef\FACT{Fact}
\gdef\PROPRT{Property}
\gdef\RESULT{Result}
\gdef\LEMMA{Lemma}
\gdef\THRM{Theorem}
\gdef\TRUE{true}
\def\thname{}
\gdef\@thm#1#2{\refstepcounter{#1}%
   \trivlist
   \@topsep \theorempreskipamount               % used by first \item
   \@topsepadd \theorempostskipamount           % used by \@endparenv
   \@ifnextchar [%
   {\@ythm{#1}{#2}}%
   {\xdef\thname{#2}
\@begintheorem{#2}{\ifx\thname\PROOF\unskip:\else\csname the#1\endcsname:\fi}\ignorespaces\ifx\Boldhead\TRUE\reset@font\itshape\fi}}%fci
\global\let\@xthm\relax
\newskip\theorempreskipamount
\newskip\theorempostskipamount
\global\setlength\theorempreskipamount{12pt plus 5pt minus 3pt}
\global\setlength\theorempostskipamount{8pt plus 3pt minus 1.5pt}
%%%%
\font\sqbox=msam10
\def\endnewtrivlist{%
  \if@inlabel
    \leavevmode
    \global \@inlabelfalse
  \fi
  \if@newlist
    \@noitemerr
    \global \@newlistfalse
  \fi
  \ifx\thname\PROOF\qquad \hfill {\sqbox\char'003}\else\fi\ifhmode\unskip \par\fi
  \if@noparlist \else
    \ifdim\lastskip >\z@
      \@tempskipa\lastskip \vskip -\lastskip
      \advance\@tempskipa\parskip \advance\@tempskipa -\@outerparskip
      \vskip\@tempskipa
    \fi
    \@endparenv
  \fi
}
%%%%
\global\let\@endtheorem=\endnewtrivlist
\@onlypreamble\@xnthm
\@onlypreamble\@ynthm
\@onlypreamble\@othm
\@onlypreamble\newtheorem
\@onlypreamble\theoremstyle
\@onlypreamble\theorembodyfont
\@onlypreamble\theoremheaderfont
\theoremstyle{plain}
%%
\ifx\upshape\undefined
\gdef\theorem@headerfont{\sc}
\else \gdef\theorem@headerfont{\normalfont\sc}\fi%bfseries
%\FACT \PROPRT \RESULT \LEMMA \THRM 

\gdef\th@plain{\normalfont%\itshape
  \def\@begintheorem##1##2{%
			\xdef\Check{##1}%
			\ifx\Check\ALGO\gdef\Boldhead{true}\else%
				\ifx\Check\CORO\gdef\Boldhead{true}\else%
					\ifx\Check\FACT\gdef\Boldhead{true}\else%
						\ifx\Check\PROPRT\gdef\Boldhead{true}\else%
							\ifx\Check\RESULT\gdef\Boldhead{true}\else%
								\ifx\Check\LEMMA\gdef\Boldhead{true}\else%
									\ifx\Check\THRM\gdef\Boldhead{true}\else%
										\gdef\Boldhead{false}\fi\fi\fi\fi\fi\fi\fi%
        \item[\hskip\parindent \theorem@headerfont\def\checkhead{##1}% \labelsep
                    \ifx\Boldhead\TRUE{\reset@font\normalfont~~\sc##1\ ##2}\enspace\else% 
					\ifx\thname\PROOF{\reset@font\sc##1:}\enspace\else##1\ ##2\enspace\fi\fi]}%
\def\@opargbegintheorem##1##2##3{%
			\xdef\Check{##1}%
			\ifx\Check\ALGO\gdef\Boldhead{true}\else%
				\ifx\Check\CORO\gdef\Boldhead{true}\else%
					\ifx\Check\FACT\gdef\Boldhead{true}\else%
						\ifx\Check\PROPRT\gdef\Boldhead{true}\else%
							\ifx\Check\RESULT\gdef\Boldhead{true}\else%
								\ifx\Check\LEMMA\gdef\Boldhead{true}\else%
									\ifx\Check\THRM\gdef\Boldhead{true}\else%
										\gdef\Boldhead{false}\fi\fi\fi\fi\fi\fi\fi%
   \item[\hskip\parindent \theorem@headerfont %labelsep
		 \ifx\Boldhead\TRUE{\reset@font\normalfont~~\sc##1\	##2}\ {\rm (##3)}{\reset@font\sc:}\enspace\else% 
					  \ifx\thname\PROOF{\reset@font\sc##1\ {\rm(##3)}:}\enspace\else%		
						   ##1\ ##2\ {\rm	(##3):\enspace}\fi\fi] 
						   \ifx\Boldhead\TRUE\itshape\fi}}
%%
\let\bm\bmath
%%
\ifSFB@referee
  \usepackage[nolists]{endfloat}
  \AtBeginDocument{\usepackage{endrotfloat}}
\fi%
%%%%%%%%%%%%%%%%%%%% NEW DIMENTIONES %%%%%%%%%%%%%
\newdimen\footheight
\footheight\z@

%\usepackage{pagefram}
%\innerskeltrue
%\textgridfalse
%%%%%
%\paperheight{10.875in}
%%\trimheight{10.875in}
\setlength\textwidth{40pc}
%
\setlength\textheight{52pc}
\addtolength\textheight{-4.65pt}
\setlength\textwidth{40pc}
%
\setlength\topmargin{54pt}%{4pc}
\setlength\oddsidemargin   {54pt}%{5pc}
\setlength\evensidemargin{4.55pc}
\setlength\headheight{7\p@}% 
\setlength\headsep   {23pt}% \typeheight - \textheight - \headheight
\setlength\topskip   {7.2\p@}
\setlength{\footskip}  {30pt} %was 26 pt
\setlength\columnsep{24\p@}
\setlength\parindent{18pt}
%%%%%%%%%%
\def\creditlinefont{\fontsize{8}{8}\selectfont\itshape}
\def\rhfont{\fontsize{10}{10}\selectfont\bfseries\mathversion{bold}}
\def\foliofont{\fontsize{10}{10}\selectfont\bfseries}
\def\thanksfont{\fontsize{8}{10}\selectfont}
\def\dropfoliofont{\fontsize{10}{10}\selectfont\bfseries}
\def\abstractfont{\fontsize{9}{11}\selectfont}%% part
\def\keywordfont{\fontsize{8}{10}\selectfont}%% part
\def\short{\fontsize{7}{7}\selectfont}
\def\authorfont{\fontsize{12}{14}\selectfont\bfseries\raggedright\mathversion{bold}}
\def\figcaptionfont{\fontsize{8}{10}\selectfont}
\def\sectionfont{\fontsize{10}{12}\selectfont\capsshape\bfseries\raggedright\mathversion{bold}}
\def\subsectionfont{\fontsize{10}{12}\selectfont\bfseries\raggedright\mathversion{bold}}
\def\subsubsectionfont{\fontsize{10}{12}\selectfont\raggedright\itshape}
%%%%%%%%%%%%%%% REFERENCES %%%%%%%%
% \usepackage[super,comma]{natbib}
% \gdef\bibnumfmt#1{#1.}
% \renewcommand\NAT@open{(}\renewcommand\NAT@close{)}
% \def\newblock{\hskip .11em plus .33em minus .07em}
% %%%%%%%%%%%%%%%%%%%%%
% \def\bibliofont{\fontsize{8}{9}\selectfont}
% \renewenvironment{thebibliography}[1]{%
% \par\vspace*{12pt} \bibsection\vspace*{4pt}\par\parindent \z@\bibpreamble\bibliofont\list
%   {\@biblabel{\arabic{NAT@ctr}}}{\@bibsetup{#1}%
%     \setcounter{NAT@ctr}{0}}%
%     \ifNAT@openbib
%       \renewcommand\newblock{\par}
%     \else
%       \renewcommand\newblock{\hskip .11em \@plus.33em \@minus.07em}%
%     \fi
%     \sloppy\clubpenalty4000\widowpenalty4000
%     \sfcode`\.=1000\relax
%     \let\citeN\cite \let\shortcite\cite
%     \let\citeasnoun\cite
%  }{\def\@noitemerr{%
%   \PackageWarning{natbib}
%      {Empty `thebibliography' environment}}%
%   \endlist\vskip-\lastskip}
% \hyphenpenalty3000
\voffset-2pc
\hoffset-5pc
%
\endinput

% End of risa.cls
