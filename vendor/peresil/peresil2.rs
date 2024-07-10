/// A location in the parsed data
trait Point: PartialOrd + Copy {
    fn zero() -> Self;
}

impl Point for usize { fn zero() -> usize { 0 } }
impl Point for i32 { fn zero() -> i32 { 0 } }

#[derive(Debug,PartialEq)]
struct Failures<P, E> {
    point: P,
    kinds: Vec<E>,
}

impl<P, E> Failures<P, E>
    where P: Point
{
    fn new() -> Failures<P, E> { Failures { point: P::zero(), kinds: Vec::new() } }

    fn add(&mut self, point: P, failure: E) {
        if point < self.point {
            // Do nothing, our existing failures are better
        } else if point > self.point {
            // The new failure is better, toss existing failures
            self.point = point;
            self.kinds.clear();
            self.kinds.push(failure);
        } else {
            // Multiple failures at the same point, tell the user all
            // the ways they could do better.
            self.kinds.push(failure);
        }
    }

    fn into_progress<T>(self) -> Progress<P, T, Vec<E>> {
        Progress { point: self.point, status: Status::Failure(self.kinds) }
    }
}

#[derive(Debug,PartialEq)]
enum Status<T, E> {
    Success(T),
    Failure(E)
}

impl<T, E> Status<T, E> {
    fn map_err<F, E2>(self, f: F) -> Status<T, E2>
        where F: FnOnce(E) -> E2
    {
        match self {
            Status::Success(x) => Status::Success(x),
            Status::Failure(x) => Status::Failure(f(x))
        }
    }
}

#[must_use]
#[derive(Debug,PartialEq)]
struct Progress<P, T, E> {
    point: P,
    status: Status<T, E>,
}

/// Success means we parsed some value, the point is at the next place
/// to start parsing Failure means we didn't parse a thing, "why" is
/// the value, the point is at the next place to start parsing (often
/// unchanged).
impl<P, T, E> Progress<P, T, E> {
    fn map_err<F, E2>(self, f: F) -> Progress<P, T, E2>
        where F: FnOnce(E) -> E2
    {
        Progress { point: self.point, status: self.status.map_err(f) }
    }

    // If we fail N optionals and then a required, it'd be nice to
    // report all the optional things. Might be difficult to do that
    // and return the optional value.
    fn optional(self, reset_to: P) -> (P, Option<T>) {
        match self {
            Progress { status: Status::Success(val), point } => (point, Some(val)),
            Progress { status: Status::Failure(..), .. } => (reset_to, None),
        }
    }
}

#[derive(Debug,PartialEq)]
struct ParseMaster<P, E> {
    failures: Failures<P, E>,
}

impl<'a, P, E> ParseMaster<P, E>
    where P: Point
{
    /// Start parsing a string
    fn new() -> ParseMaster<P, E> {
        ParseMaster {
            failures: Failures::new(),
        }
    }

    /// consume a potential error
    /// used when the last step is sequential
    fn consume<T>(&mut self, progress: Progress<P, T, E>) -> Progress<P, T, ()> {
        match progress {
            Progress { status: Status::Success(..), .. } => progress.map_err(|_| ()),
            Progress { status: Status::Failure(f), point } => {
                self.failures.add(point, f);
                Progress { status: Status::Failure(()), point: point }
            }
        }
    }

    /// run a single subparser
    // TODO: decide utility
    fn require<F, T>(&mut self, parser: F) -> Progress<P, T, ()>
        where F: FnOnce(&mut ParseMaster<P, E>) -> Progress<P, T, E>
    {
        self.alternate()
            .one(parser)
            .finish()
    }

    /// run sub-parsers in order until one succeeds
    fn alternate<'pm, T>(&'pm mut self) -> Alternate<'pm, P, T, E> {
        Alternate {
            master: self,
            current: None,
        }
    }

    /// Runs the parser until it fails. Each successfully parsed value
    /// is kept and returned. This always succeeds, but the
    /// point may be left in the middle of a token.
    // TODO: perhaps we need a concept of "fully parsed" - that would allow this to fail
    // TODO: Maybe we should return the parsing point from the beginning of the loop
    fn zero_or_more<F, T>(&mut self, point: P, mut parser: F) -> Progress<P, Vec<T>, ()>
        where F: FnMut(&mut ParseMaster<P, E>, P) -> Progress<P, T, E>
    {
        let mut current_point = point;
        let mut values = Vec::new();

        loop {
            let progress = parser(self, current_point);
            match progress {
                Progress { status: Status::Success(v), point } => {
                    values.push(v);
                    current_point = point;
                },
                Progress { status: Status::Failure(f), point } => {
                    self.failures.add(point, f);
                    current_point = point;
                    break;
                }
            }
        }

        Progress { status: Status::Success(values), point: current_point }
    }

    /// When all parsing is complete, regain access to the failures
    fn finish<T>(self, progress: Progress<P, T, ()>) -> Progress<P, T, Vec<E>> {
        match progress {
            Progress { status: Status::Success(..), .. } => progress.map_err(|_| Vec::new()),
            Progress { status: Status::Failure(..), .. } => self.failures.into_progress(),
        }
    }
}

#[must_use]
struct Alternate<'pm, P : 'pm, T, E : 'pm> {
    master: &'pm mut ParseMaster<P, E>,
    current: Option<Progress<P, T, ()>>,
}

/// An alternate consumes the error of children, tracking them
impl<'pm, P, T, E> Alternate<'pm, P, T, E>
    where P: Point
{
    fn run_one<F>(&mut self, parser: F)
        where F: FnOnce(&mut ParseMaster<P, E>) -> Progress<P, T, E>
    {
        let r = parser(self.master);
        let r = self.master.consume(r);
        self.current = Some(r);
    }

    fn one<F>(mut self, parser: F) -> Alternate<'pm, P, T, E>
        where F: FnOnce(&mut ParseMaster<P, E>) -> Progress<P, T, E>
    {
        match self.current {
            None => self.run_one(parser),
            Some(Progress { status: Status::Success(..), .. }) => {},
            Some(Progress { status: Status::Failure(..), .. }) => self.run_one(parser),
        }

        self
    }

    fn finish(self) -> Progress<P, T, ()> {
        self.current.unwrap()
    }
}

macro_rules! try_parse(
    ($e:expr) => ({
        match $e {
            Progress { status: Status::Success(val), point } => (point, val),
            Progress { status: Status::Failure(val), point } => {
                return Progress { point: point, status: Status::Failure(val) }
            }
        }
    });
);

#[derive(Debug,Copy,Clone,PartialEq,PartialOrd)]
struct StringPoint<'a> {
    s: &'a str,
    p: usize,
}

impl<'a> Point for StringPoint<'a> {
    fn zero() -> StringPoint<'a> { StringPoint { s: "", p: 0} }
}

impl<'a> StringPoint<'a> {
    fn new(s: &'a str) -> StringPoint<'a> {
        StringPoint { s: s, p: 0 }
    }

    fn consume_literal(self, val: &str) -> StringProgress<'a, &'a str> {
        if self.s.starts_with(val) {
            let len = val.len();
            let matched = &self.s[..len];
            let rest = &self.s[len..];
            Progress { point: StringPoint { s: rest, p: self.p + len }, status: Status::Success(matched) }
        } else {
            Progress { point: self, status: Status::Failure(()) }
        }
    }
}

////////////////////////////////////////

#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord)]
struct AnError(u8);

type SimpleMaster = ParseMaster<usize, AnError>;
type SimpleProgress<T> = Progress<usize, T, AnError>;

#[test]
fn one_error() {
    let mut d = ParseMaster::new();

    let r = d.require::<_, ()>(|_| Progress { point: 0, status: Status::Failure(AnError(1)) });

    let r = d.finish(r);

    assert_eq!(r, Progress { point: 0, status: Status::Failure(vec![AnError(1)]) });
}

#[test]
fn two_error_at_same_point() {
    let mut d = ParseMaster::new();

    let r = d.alternate::<()>()
        .one(|_| Progress { point: 0, status: Status::Failure(AnError(1)) })
        .one(|_| Progress { point: 0, status: Status::Failure(AnError(2)) })
        .finish();

    let r = d.finish(r);

    assert_eq!(r, Progress { point: 0, status: Status::Failure(vec![AnError(1), AnError(2)]) });
}

#[test]
fn first_error_is_better() {
    let mut d = ParseMaster::new();

    let r = d.alternate::<()>()
        .one(|_| Progress { point: 1, status: Status::Failure(AnError(1)) })
        .one(|_| Progress { point: 0, status: Status::Failure(AnError(2)) })
        .finish();

    let r = d.finish(r);

    assert_eq!(r, Progress { point: 1, status: Status::Failure(vec![AnError(1)]) });
}

#[test]
fn second_error_is_better() {
    let mut d = ParseMaster::new();

    let r = d.alternate::<()>()
        .one(|_| Progress { point: 0, status: Status::Failure(AnError(1)) })
        .one(|_| Progress { point: 1, status: Status::Failure(AnError(2)) })
        .finish();

    let r = d.finish(r);

    assert_eq!(r, Progress { point: 1, status: Status::Failure(vec![AnError(2)]) });
}

#[test]
fn one_success() {
    let mut d = ParseMaster::<_, ()>::new();

    let r = d.require(|_| Progress { point: 0, status: Status::Success(42) });

    let r = d.finish(r);
    assert_eq!(r, Progress { point: 0, status: Status::Success(42) });
}

#[test]
fn success_after_failure() {
    let mut d = ParseMaster::new();

    let r = d.alternate()
        .one(|_| Progress { point: 0, status: Status::Failure(AnError(1)) })
        .one(|_| Progress { point: 0, status: Status::Success(42) })
        .finish();

    let r = d.finish(r);
    assert_eq!(r, Progress { point: 0, status: Status::Success(42) });
}

#[test]
fn success_before_failure() {
    let mut d = ParseMaster::<_, ()>::new();

    let r = d.alternate()
        .one(|_| Progress { point: 0, status: Status::Success(42) })
        .one(|_| panic!("Should not even be called"))
        .finish();

    let r = d.finish(r);
    assert_eq!(r, Progress { point: 0, status: Status::Success(42) });
}

#[test]
fn sequential_success() {
    fn first(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
        Progress { point: pt + 1, status: Status::Success(1) }
    }

    fn second(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
        Progress { point: pt + 1, status: Status::Success(2) }
    }

    fn both(d: &mut SimpleMaster, pt: usize) -> SimpleProgress<(u8,u8)> {
        let (pt, val1) = try_parse!(first(d, pt));
        let (pt, val2) = try_parse!(second(d, pt));
        Progress { point: pt, status: Status::Success((val1, val2)) }
    }

    let mut d = ParseMaster::new();
    let r = both(&mut d, 0);
    let r = d.consume(r);
    let r = d.finish(r);

    assert_eq!(r, Progress { point: 2, status: Status::Success((1,2)) });
}

#[test]
fn child_parse_succeeds() {
    fn parent(d: &mut SimpleMaster, pt: usize) -> SimpleProgress<(u8,u8)> {
        let (pt, val1) = try_parse!(child(d, pt));
        Progress { point: pt + 1, status: Status::Success((val1, 2)) }
    }

    fn child(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
        Progress { point: pt + 1, status: Status::Success(1) }
    }

    let mut d = ParseMaster::new();
    let r = parent(&mut d, 0);
    let r = d.consume(r);
    let r = d.finish(r);

    assert_eq!(r, Progress { point: 2, status: Status::Success((1, 2)) });
}

#[test]
fn child_parse_fails_child_step() {
    fn parent(d: &mut SimpleMaster, pt: usize) -> SimpleProgress<(u8,u8)> {
        let (pt, val1) = try_parse!(child(d, pt));
        Progress { point: pt + 1, status: Status::Success((val1, 2)) }
    }

    fn child(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
        Progress { point: pt + 1, status: Status::Failure(AnError(1)) }
    }

    let mut d = ParseMaster::new();
    let r = parent(&mut d, 0);
    let r = d.consume(r);
    let r = d.finish(r);

    assert_eq!(r, Progress { point: 1, status: Status::Failure(vec![AnError(1)]) });
}

#[test]
fn child_parse_fails_parent_step() {
    fn parent(d: &mut SimpleMaster, pt: usize) -> SimpleProgress<(u8,u8)> {
        let (pt, _) = try_parse!(child(d, pt));
        Progress { point: pt + 1, status: Status::Failure(AnError(2)) }
    }

    fn child(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
        Progress { point: pt + 1, status: Status::Success(1) }
    }

    let mut d = ParseMaster::new();
    let r = parent(&mut d, 0);
    let r = d.consume(r);
    let r = d.finish(r);

    assert_eq!(r, Progress { point: 2, status: Status::Failure(vec![AnError(2)]) });
}

#[test]
fn alternate_with_children_parses() {
    fn first(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
        Progress { point: pt + 1, status: Status::Failure(AnError(1)) }
    }

    fn second(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
        Progress { point: pt + 1, status: Status::Success(1) }
    }

    fn both(d: &mut SimpleMaster, pt: usize) -> Progress<usize, u8, ()> {
        d.alternate()
            .one(|d| first(d, pt))
            .one(|d| second(d, pt))
            .finish()
    }

    let mut d = ParseMaster::new();
    let r = both(&mut d, 0);
    let r = d.finish(r);

    assert_eq!(r, Progress { point: 1, status: Status::Success(1) });
}

#[test]
fn optional_present() {
    fn optional(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
        Progress { point: pt + 1, status: Status::Success(1) }
    }

    let mut d = ParseMaster::new();
    let (pt, val) = optional(&mut d, 0).optional(0);

    assert_eq!(pt, 1);
    assert_eq!(val, Some(1));
}

#[test]
fn optional_missing() {
    fn optional(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
        Progress { point: pt + 1, status: Status::Failure(AnError(1)) }
    }

    let mut d = ParseMaster::new();
    let (pt, val) = optional(&mut d, 0).optional(0);

    assert_eq!(pt, 0);
    assert_eq!(val, None);
}

#[test]
fn zero_or_more() {
    let mut remaining: u8 = 2;

    let mut body = |_: &mut SimpleMaster, pt: usize| -> SimpleProgress<u8> {
        if remaining > 0 {
            remaining -= 1;
            Progress { point: pt + 1, status: Status::Success(remaining) }
        } else {
            Progress { point: pt + 1, status: Status::Failure(AnError(1)) }
        }
    };

    let mut d = ParseMaster::new();
    let r = d.zero_or_more(0, |d, pt| body(d, pt));

    assert_eq!(r, Progress { point: 3, status: Status::Success(vec![1, 0]) });
}

type StringMaster<'a> = ParseMaster<StringPoint<'a>, AnError>;
type StringProgress<'a, T> = Progress<StringPoint<'a>, T, ()>;

#[test]
fn string_sequential() {
    fn all<'a>(pt: StringPoint<'a>) -> StringProgress<'a, (&'a str, &'a str, &'a str)> {
        let (pt, a) = try_parse!(pt.consume_literal("a"));
        let (pt, b) = try_parse!(pt.consume_literal("b"));
        let (pt, c) = try_parse!(pt.consume_literal("c"));

        Progress {point: pt, status: Status::Success((a,b,c))}
    }

    let mut d = ParseMaster::new();
    let pt = StringPoint::new("abc");

    let r = all(pt);
    let r = d.consume(r);
    let r = d.finish(r);

    assert_eq!(r, Progress { point: StringPoint { s: "", p: 3 }, status: Status::Success(("a", "b", "c")) });
}

#[test]
fn string_alternate() {
    fn any<'a>(d: &mut StringMaster<'a>, pt: StringPoint<'a>) -> StringProgress<'a, &'a str> {
        d.alternate()
            .one(|_| pt.consume_literal("a").map_err(|_| AnError(1)))
            .one(|_| pt.consume_literal("b").map_err(|_| AnError(2)))
            .one(|_| pt.consume_literal("c").map_err(|_| AnError(3)))
            .finish()
    }

    let mut d = ParseMaster::new();
    let pt = StringPoint::new("c");

    let r = any(&mut d, pt);
    let r = d.finish(r);

    assert_eq!(r, Progress { point: StringPoint { s: "", p: 1 }, status: Status::Success("c") });
}

#[test]
fn string_zero_or_more() {
    fn any<'a>(d: &mut StringMaster<'a>, pt: StringPoint<'a>) -> StringProgress<'a, Vec<&'a str>> {
        d.zero_or_more(pt, |_, pt| pt.consume_literal("a").map_err(|_| AnError(1)))
    }

    let mut d = ParseMaster::new();
    let pt = StringPoint::new("aaa");

    let r = any(&mut d, pt);
    let r = d.finish(r);

    assert_eq!(r, Progress { point: StringPoint { s: "", p: 3 }, status: Status::Success(vec!["a", "a", "a"]) });
}
