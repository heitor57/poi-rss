--DROP DATABASE yelp;
CREATE DATABASE yelp;
use yelp;

CREATE TABLE business (
business_id varchar(22),
name text,
address text,
city text,
state text,
`postal code` text,
latitude float,
longitude float,
stars float,
review_count int,
is_open int,-- attributes not using yet
PRIMARY KEY (business_id)
);

CREATE TABLE user (
user_id varchar(22),
name text,
review_count int,
yelping_since text,--  "friends": [
useful int,
funny int,
cool int,
fans int,-- elite : [
average_stars float,
compliment_hot int,
compliment_more int,
compliment_profile int,
compliment_cute int,
compliment_list int,
compliment_note int,
compliment_plain int,
compliment_cool int,
compliment_funny int,
compliment_writer int,
compliment_photos int,
PRIMARY KEY (user_id)
);

CREATE TABLE review (
review_id varchar(22),
user_id varchar(22),
business_id varchar(22),
stars int,
date text,
text text,
useful int,
funny int,
cool int,
PRIMARY KEY(review_id),
CONSTRAINT fk_review_user_id FOREIGN KEY (user_id) REFERENCES user(user_id),
CONSTRAINT fk_review_business_id FOREIGN KEY (business_id) REFERENCES business(business_id)
);


CREATE TABLE checkin (
business_id varchar(22),
date longtext,
PRIMARY KEY(business_id),
CONSTRAINT fk_checkin_business_id FOREIGN KEY (business_id) REFERENCES business(business_id)
);


CREATE TABLE tip (
text text,
date text,
compliment_count int,
business_id varchar(22),
user_id varchar(22),
CONSTRAINT fk_tip_business_id FOREIGN KEY (business_id) REFERENCES business(business_id),
CONSTRAINT fk_tip_user_id FOREIGN KEY (user_id) REFERENCES user(user_id)
);

CREATE TABLE photo (
photo_id varchar(22),
business_id varchar(22),
caption text,
label text,
PRIMARY KEY(photo_id),
CONSTRAINT fk_photo_business_id FOREIGN KEY (business_id) REFERENCES business(business_id)
);

CREATE TABLE friend (
       user_id1 varchar(22),
       user_id2 varchar(22),
       PRIMARY KEY (user_id1,user_id2),
       CONSTRAINT fk_friend_user_id1 FOREIGN KEY (user_id1) REFERENCES user(user_id),
       CONSTRAINT fk_friend_user_id2 FOREIGN KEY (user_id2) REFERENCES user(user_id)
);

CREATE TABLE elite (
       user_id varchar(22),
       year int,
       PRIMARY KEY (user_id,year),
       CONSTRAINT fk_elite_user_id FOREIGN KEY (user_id) REFERENCES user(user_id)
);

CREATE TABLE category (
       business_id varchar(22),
       category_name text,
       CONSTRAINT fk_category_business_id FOREIGN KEY (business_id) REFERENCES business(business_id)
);
