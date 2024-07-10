//! # Simplistic sentence parser
//!
//! This example shows how parsing can be split into separate
//! tokenization and parsing steps.
//!
//! ## Grammar
//!
//! Document := Sentence* ε
//! Sentence := Article S Noun S Verb "."
//! Article  := "a" | "the"
//! Noun     := "dog" | "cat" | "bird"
//! Verb     := "runs" | "eats" | "sleeps"
//! S        := " "+

#[macro_use]
extern crate peresil;

use peresil::v2::{self, ParseMaster, StringPoint};

#[derive(Debug,Copy,Clone,PartialEq)]
enum Article { A, The }
#[derive(Debug,Copy,Clone,PartialEq)]
enum Noun { Dog, Cat, Bird }
#[derive(Debug,Copy,Clone,PartialEq)]
enum Verb { Runs, Eats, Sleeps }

#[derive(Debug,Copy,Clone,PartialEq)]
struct Sentence {
    article: Article,
    noun: Noun,
    verb: Verb,
}

#[derive(Debug,Copy,Clone,PartialEq)]
enum Error {}

type TokenMaster<'a> = ParseMaster<StringPoint<'a>, Error>;
type TokenProgress<'a, T> = v2::Progress<StringPoint<'a>, T, Error>;

fn consume_article<'a>(pm: &mut TokenMaster<'a>, pt: StringPoint<'a>) -> TokenProgress<'a, Article> {
    v2::Progress::failure()
}

fn consume_noun<'a>(pm: &mut TokenMaster<'a>, pt: StringPoint<'a>) -> TokenProgress<'a, Article> {
    v2::Progress::failure()
}

fn consume_verb<'a>(pm: &mut TokenMaster<'a>, pt: StringPoint<'a>) -> TokenProgress<'a, Article> {
    v2::Progress::failure()
}

fn consume_article<'a>(pm: &mut TokenMaster<'a>, pt: StringPoint<'a>) -> TokenProgress<'a, Article> {
    v2::Progress::failure()
}

fn consume_article<'a>(pm: &mut TokenMaster<'a>, pt: StringPoint<'a>) -> TokenProgress<'a, Article> {
    v2::Progress::failure()
}

fn consume_article<'a>(pm: &mut TokenMaster<'a>, pt: StringPoint<'a>) -> TokenProgress<'a, Article> {
    v2::Progress::failure()
}

fn tokenize(s: &str) {
    let mut pm = ParseMaster::new();
    let pt = StringPoint::new(s);

    pm.zero_or_more(pt, |pm, pt| {
        pm.alternate()
            .one(|pm| consume_article(pm, pt))
            .one(|pm| consume_noun(pm, pt))
            .one(|pm| consume_verb(pm, pt))
            .one(|pm| consume_full_stop(pm, pt))
            .one(|pm| consume_whitespace(pm, pt))
            .one(|pm| consume_end_of_input(pm, pt))
            .finish()
    })
}

fn parse(s: &str) -> Result<Vec<Sentence>, (usize, Vec<Error>)> {
    let tokens = tokenize(s);
    Err((0, vec![]))
}

#[test]
fn a() {
    assert_eq!(
        parse("the dog runs."),
        Ok(vec![Sentence { article: Article::The, noun: Noun::Dog, verb: Verb::Runs }])
    );
}
