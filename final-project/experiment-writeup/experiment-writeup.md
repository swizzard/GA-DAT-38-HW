# Experiment Writeup

Pitchers in Major League Baseball vary their grips, motions, and points of release
when they throw the ball. These various combinations are generally grouped into
pitch types, of which there are two major families:
* Fastballs, which rely more on speed than movement, and
* Offspeed pitches, which rely more on movement than speed
Different types of pitches can differ wildly in speed and horizontal and vertical
movement, presumably requiring different reactions from batters. While some research
has been done on pitch sequencing [by type][1] and [by location][2], this project
will attempt to discern whether there are statistically significant differences
between batters in how successful they are when faced with different kinds of pitches.

Baseball statistics have been around for over a [century][3]; the introduction
of [PITCHf/x][4] in 2006 allows for pitch-by-pitch analysis, when previously the
at-bat was the smallest available analytical unit.

PITCHf/x data is [available publically][5], from MLB Advanced Media, in the form
of XML documents. [Brooks Baseball][6] maintain their own version as well. I wrote
[my own package][7] to scrape the MLBAM data, and used an [open-source scraper][8]
to retrieve the Brooks data.

The data contains [over 20 fields][9], which encode everything from pitcher and
batter ids, to pitch speed, to `x`, `y`, and `z` coordinates of the ball as it
passes over home plate. MLBAM uses an algorithm to classify pitches based on
this data; the pitch type is included in the data as one of roughly sixteen
[two-letter codes][10]. Brooks Baseball manually verifies and adjusts the
trajectory and pitch type data they provide, although their data extends back
only to 2010, while substantially-complete data is available from MLBAM dating
to 2008. The data also contains a field called `type`, which indicates the
result of the pitch, encoded as one of three values:
* `B` indicates a ball, i.e. a pitch thrown outside the designated strike zone
* `S` indicates a strike, i.e. a pitch inside the strike zone that the batter either missed or did not swing at
* `X` indicates a hit
We can thus derive batter and pitcher performance by comparing the ratio of hits
to non-hits, aggregating across pitch types and other variables.

Baseball is generally high-variance, and while there is a lot of data available,
there are numerous potential confounds, including but not limited to overall
batter and pitcher skill, undocumented injuries, and the weather. Ideally, the
volume of data and careful analysis can mitigate the impact of these factors, but there
are no firm guarantees. There is similarly no guarantee that pitch types are significant,
nor that batters perform consistantly against different types of pitches.

Ultimately, the goal of this project is to advance the state of [Sabermetric][11] research,
and provide fans (and perhaps players and managers) more insight into the game. Depending
on the results of the statistical analysis, it might be possible to build a model to
predict batter performance against various types of pitches, which could be used
by both casual fans and those in the Fantasy Baseball community.


[1]: http://www.beyondtheboxscore.com/2013/7/26/4558940/strikeout-pitch-sequences-pitchfx-sabermetrics
[2]: http://www.beyondtheboxscore.com/2013/8/9/4599550/strikeout-pitch-sequences-by-location-pitchfx-sabermetrics
[3]: https://en.wikipedia.org/wiki/Baseball_statistics
[4]: https://en.wikipedia.org/wiki/PITCHf/x
[5]: http://gd2.mlb.com/components/game/mlb/
[6]: http://www.brooksbaseball.net/
[7]: https://github.com/swizzard/voros
[8]: https://github.com/mattdennewitz/mlb-brooks-pitch-importer
[9]: https://fastballs.wordpress.com/2007/08/02/glossary-of-the-gameday-pitch-fields/
[10]: http://www.fangraphs.com/library/pitch-type-abbreviations-classifications/
[11]: https://en.wikipedia.org/wiki/Sabermetrics
