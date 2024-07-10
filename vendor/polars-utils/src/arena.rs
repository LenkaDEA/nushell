use crate::error::*;
use crate::slice::GetSaferUnchecked;

unsafe fn index_of_unchecked<T>(slice: &[T], item: &T) -> usize {
    (item as *const _ as usize - slice.as_ptr() as usize) / std::mem::size_of::<T>()
}

fn index_of<T>(slice: &[T], item: &T) -> Option<usize> {
    debug_assert!(std::mem::size_of::<T>() > 0);
    let ptr = item as *const T;
    unsafe {
        if slice.as_ptr() < ptr && slice.as_ptr().add(slice.len()) > ptr {
            Some(index_of_unchecked(slice, item))
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct Node(pub usize);

impl Default for Node {
    fn default() -> Self {
        Node(usize::MAX)
    }
}

#[derive(Clone)]
pub struct Arena<T> {
    items: Vec<T>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple Arena implementation
/// Allocates memory and stores item in a Vec. Only deallocates when being dropped itself.
impl<T> Arena<T> {
    pub fn add(&mut self, val: T) -> Node {
        let idx = self.items.len();
        self.items.push(val);
        Node(idx)
    }

    pub fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn new() -> Self {
        Arena { items: vec![] }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Arena {
            items: Vec::with_capacity(cap),
        }
    }

    pub fn get_node(&self, val: &T) -> Option<Node> {
        index_of(&self.items, val).map(Node)
    }

    pub fn swap(&mut self, idx_a: Node, idx_b: Node) {
        self.items.swap(idx_a.0, idx_b.0)
    }

    #[inline]
    pub fn get(&self, idx: Node) -> &T {
        self.items.get(idx.0).unwrap()
    }

    #[inline]
    /// # Safety
    /// Doesn't do any bound checks
    pub unsafe fn get_unchecked(&self, idx: Node) -> &T {
        self.items.get_unchecked_release(idx.0)
    }

    #[inline]
    pub fn get_mut(&mut self, idx: Node) -> &mut T {
        self.items.get_mut(idx.0).unwrap()
    }

    #[inline]
    pub fn replace(&mut self, idx: Node, val: T) {
        let x = self.get_mut(idx);
        *x = val;
    }
    pub fn clear(&mut self) {
        self.items.clear()
    }
}

impl<T: Clone> Arena<T> {
    pub fn duplicate(&mut self, node: Node) -> Node {
        let item = self.items[node.0].clone();
        self.add(item)
    }
}

impl<T: Default> Arena<T> {
    #[inline]
    pub fn take(&mut self, idx: Node) -> T {
        std::mem::take(self.get_mut(idx))
    }

    pub fn replace_with<F>(&mut self, idx: Node, f: F)
    where
        F: FnOnce(T) -> T,
    {
        let val = self.take(idx);
        self.replace(idx, f(val));
    }

    pub fn try_replace_with<F>(&mut self, idx: Node, mut f: F) -> Result<()>
    where
        F: FnMut(T) -> Result<T>,
    {
        let val = self.take(idx);
        self.replace(idx, f(val)?);
        Ok(())
    }
}
