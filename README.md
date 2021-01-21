\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx} %package to manage images
\usepackage{array}
\usepackage{tabu}
\graphicspath{ {images/} }
\title{Machine Learning Report}
\author{Joshua Nuttall R00128796}
\date{November 2017}

% Definition of \maketitle
\makeatletter         
\def\@maketitle{
\includegraphics[width =\textwidth]{CITCREST.jpg}\\[8ex]
\begin{center}
{\Huge \bfseries \sffamily \@title }\\[4ex] 
{\Large  \@author}\\[4ex] 
\@date\\[8ex]
\end{center}}
\makeatother

\begin{document}

\maketitle
\newpage
\tableofcontents
\clearpage
\section{Introduction}
\subsection{Abstract}
A comparison on different model performances at classifying organizations responsible for acts of terrorism between the period of 1970 and 2016. Using approaches such as K-Fold validation, K-nearest neighbour, naive Bayes, decision tree's and Random Forests to achieve as high an accuracy as possible.


\subsection{Motivation and Data Set}
In modern times, globally countries are consistently under the threat of terrorism. While becoming more and more frequent in western countries such as the UK where it has experienced 5 acts of terrorism in 2017 alone, comparing that to 0 terrorist attacks in 2016 and 1 in 2015. While these numbers are alarming to western culture they are nothing compared to the weekly attacks committed in developing countries in regions such as Africa and the middle east where current tensions, civil unrest and extremist views breed the ingredients for terrorist organizations to be created. To aid in deterring future terrorist attacks it is important to identify the culprit accurately and fast. To that end we are comparing different classifiers to examine which is more applicable to the Global Terrorism Database.

The GTD is an open source database updated annually dating from 1970 to 2016, It contains over 170,000 incidents of which all are vetted and pulled from credible sources.

\subsection{Designated Approach}
The initial approach is to remove all unnecessary fields, further more remove all rows that do not have a terrorist organization responsible for the incident. Implement one hot encoding on remaining features such as time/date and string features. Next run the remaining data through a range of classifiers using k-fold validation to return an average accuracy for each, selecting the one with greater accuracy we further implement this classifier to be used on the test data.
\newpage

\section{Current Research}
The primary goal of this project was to compare models for use in classifying terrorist attacks by organization. In short the aim was to predict the organization responsible for an attack, to that extent  researching specifically that subject returned very little documents so I had to broaden my work area to be relevant. Looking at previous papers on machine learning in relation to terrorism and the use of the GTD (Global terrorism database) I come across "Understanding the dynamics of terrorism events with multiple-discipline data sets and machine learning approach", by Ding F, Ge Q, Jiang D, Fu J and Hao M \cite{ml-gtd}, the contents of which provides great incite into machine learning for use in predictive threat analysis with a given region. It accomplishes this by using environmental factors (famine, drought etc..) accompanied with historical data from the GTD, crime data, Ethnic diversity and recourse's available to a given region to predict the threat level of an attack in a region in a given time.
\newline\newline
After further investigation relative material to the objective at hand was found in a paper labeled "Terrorist Group Prediction Using Data Classification" By Faryal Gohar, Wasi Haider Butt and Usman Qamar \cite{sim-gtd}. In this paper the team of 3 investigate the use of a majority vote system for use in predicting organizations responsible for acts of terrorism. Similar to me they have chosen 4 models for comparison Naive Bayes K-Nearest Neighbour, Decision Stump and ID3. There take, took an approach of selecting the attributes month, city, country, weapon type, attack type, target and group name on the basis being these are most relative to deciding a group similarly how my features where selected based on this reasoning. In comparison however my data was run against the initial 1400 organization and then the top 20 most frequent oppose to there approach of running against the top 6 most frequent. Further mention of there research will occur in comparison to my results below.


\section{Implementation}

\subsection{Pre-Processing Stage 1}
My first approach of pre-processing the data was to remove all organization that had committed less than 5 attacks, drop all rows apart from target nationality, country, region, weapon type, attack type and dates this resulted in X amount of rows. I further used one hot encoding on features such as country, nationality, weapon type and attack type. Running this through models such as K-Nearest Neighbour, Naive Bayes, Random Forrest and Decision Tree returned results as follows:

\begin{center}
    \begin{tabular}{ |m{31em}|m{4em}|} 
        \hline
        \multicolumn{2}{|c|}{Results Var 1} \\\hline
        Row size   &  85862\\ \hline
        feature size   &  387\\ \hline
        K-Nearest Neighbour   &  63.1\\ \hline
        Naive Bayes &  61.1  \\ \hline
        Random Forrest &  73.3  \\  \hline
        Decision Tree &  72.2  \\\hline
    \end{tabular}
\end{center}

\subsection{Pre-Processing Stage 2}

I found it hard to improve the accuracy's received above by removing or adding features, I decided to remove a large portion of my organization and limited it to the 20 most frequent organizations. The results where as follows:

\begin{center}
    \begin{tabular}{ |m{31em}|m{4em}|} 
        \hline
        \multicolumn{2}{|c|}{Results Var 2} \\\hline
        Row size   &  46810\\ \hline
        feature size   &  387\\ \hline
        K-Nearest Neighbour   &  89.7\\ \hline
        Naive Bayes &  92.9  \\ \hline
        Random Forrest &  95.4  \\  \hline
        Decision Tree &  95.2  \\\hline
    \end{tabular}
\end{center}

\subsection{Pre-Processing Stage 2 (Alternative)}
Alternatively to using one hot encoding I investigated the use of label encoding for purposes of code optimization whilst run time was greatly increased (5 seconds oppose to 1 to 2 minutes) almost all values upon an initial test remained the same except for Naive Bayes which was entirely expected due to my implementation of a Bernoulli Naive Bayes formula oppose to Gaussian meaning I relied on the binary data generated from one hot encoding, the results are as follows:

\begin{center}
    \begin{tabular}{ |m{31em}|m{4em}|} 
        \hline
        \multicolumn{2}{|c|}{Results Var 3} \\\hline
        Row size   &  46810\\ \hline
        feature size   &  10\\ \hline
        K-Nearest Neighbour   &  92.4\\ \hline
        Naive Bayes &  32.3  \\ \hline
        Random Forrest &  95.9  \\  \hline
        Decision Tree &  95.1  \\\hline
    \end{tabular}
\end{center}

Since a Bernoulli naive Bayes formula is being used incorrectly in this manor I included a Gaussian variant producing the following results:

\begin{center}
    \begin{tabular}{ |m{31em}|m{4em}|} 
        \hline
        \multicolumn{2}{|c|}{Results Var 4} \\\hline
        Row size   &  46810\\ \hline
        feature size   &  10\\ \hline
        K-Nearest Neighbour   &  92.4\\ \hline
        Naive Bayes &  32.3  \\ \hline
        Gaussian Naive Bayes &  84.1  \\ \hline
        Random Forrest &  95.9  \\  \hline
        Decision Tree &  95.1  \\\hline
    \end{tabular}
\end{center}

\subsection{Analyzing Results}
From the results seen above and how they have improved over each change to the implementation we can clearly see a trend being that Random Forrest produces the most accurate results out of each model used. As mentioned above a team of 3 has run similar tests and will act as a good comparison against my results presented, important to remember that my data was run against almost 4 times as much groups as the comparing data which in turn should reduce the accuracy.
\newline
\begin{center}
    \begin{tabu} to 0.8\textwidth { | X[c] | X[c] | X[c] | }
        \hline
        \multicolumn{3}{|c|}{Comparison} \\
        \hline
        feature &  foreign & home\\ 
        \hline
        classes & 6 & 20\\ 
        \hline
        feature size & 8 &  10\\ 
        \hline
        K-Nearest Neighbour & 83.4 &  92.4\\ 
        \hline
        Gaussian Naive Bayes & 92.7 &  84.1  \\ 
        \hline
        Random Forrest & - &  95.9  \\  
        \hline
        Decision Tree & 84.9 &  95.1  \\
        \hline
        Decision Stump & 91.3 & - \\ 
        \hline
    \end{tabu}
\end{center}
We can clearly see a difference of over 10 percent which I link directly to the features I have used vs the features they have used. Although I have used 2 more the relevance of my selected features are greater than there's such as region, country of attack and nationality of target oppose to just using the city. This matters as certain organizations may only attack or predominately attack certain countries for example ISIL may usually attack middle eastern countries but occasionally attack western countries, alternatively IRA may predominately attack the UK or UK nationals abroad. Equally important that the comparing report has failed to account for is the year of attack. An explanation of its relevance can be thought of when a group was most active, throughout the 1970's occurred the majority of IRA attacks in the western European region, dying down to a handful post late 1990's. This tells us that a group attacking in western Europe of country being UK or Ireland during the year of 197X was most likely the IRA oppose to the same scenario with year being 201X which may now be ISIL.
\newline
Taking from our results and using the Random Forest we can produce a confusion matrix as follows and giving us an accuracy of 96.1 percent:
\newline
\includegraphics[width =\textwidth]{foo.png}\\[8ex]
\newline
\newpage
\section{Conclusion}
Comparing the different methods and models was an intriguing and a fun exercise that resulted in interesting results. While one hot encoding can be hugely beneficial, from playing with multiple configurations and features such as using the method on year/month/day since there is so much data produced it was impossible to run any sort of model due to the amount of processing required as a result I've learned of the taxation one hot can take on a system when used on big data features. On the contrary while there was a slight accuracy loss (negligible) when using label encoding it almost certainly makes up for this with the performance increase it offers, mainly due to not increasing the feature size. Another result that surprised me was the difference in accuracy or lack there of between decision tree and random forest models. While a random forest is nothing more than a collection of decision tree's one would expect far greater accuracy oppose to one single decision tree with only a different of less than 1 percent. However, maybe this was down to the data in use.

\begin{thebibliography}{9}

    \bibitem{ml-gtd}
    Ding F, Ge Q, Jiang D, Fu J, Hao M (2017)
    Understanding the dynamics of terrorism events
    with multiple-discipline datasets and machine
    learning approach. PLoS ONE 12(6): e0179057.
    
    \bibitem{sim-gtd}
    Terrorist Group Prediction Using Data Classification
    Faryal Gohar, Wasi Haider Butt, Usman Qamar
    Department of Computer Engineering, College of Electrical and Mechanical Engineering
    National University of Sciences and Technology
    Rawalpindi, Pakistan (2014)
    
\end{thebibliography}
\end{document}
