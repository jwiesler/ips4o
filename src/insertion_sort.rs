use std::cmp::Ordering;

pub fn is_sorted_by<T, F>(values: &[T], is_less: &F) -> bool
where
    F: Fn(&T, &T) -> bool,
{
    let ord = |a, b| {
        if is_less(a, b) {
            Ordering::Less
        } else if is_less(b, a) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    };

    values
        .windows(2)
        .all(|w| ord(&w[0], &w[1]) != Ordering::Greater)
}

pub fn sort<T, F>(values: &mut [T], is_less: &F)
where
    T: Copy,
    F: Fn(&T, &T) -> bool,
{
    for i in 1..values.len() {
        let value = values[i];
        if is_less(&value, &values[0]) {
            // copy everything to the right by 1
            values.copy_within(0..i, 1);
            values[0] = value;
        } else {
            // make space
            let mut cur = i;
            let mut next = i - 1;
            while is_less(&value, &values[next]) {
                values[cur] = values[next];
                cur = next;
                next -= 1;
            }
            // place the value
            values[cur] = value;
        }
    }
}

#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use super::{is_sorted_by, sort};

    fn test_with<T: Ord + Copy + Debug>(values: &mut [T]) {
        let is_less = |a: &T, b: &T| a.lt(b);
        sort(values, &is_less);
        assert!(
            is_sorted_by(values, &is_less),
            "{:?} should be sorted",
            values
        );
    }

    #[test]
    fn test_sort() {
        test_with(&mut [0; 0usize]);
        test_with(&mut [0]);
        test_with(&mut [3, 2, 1, 0]);
        test_with(&mut [0, 1, 2, 3]);
        test_with(&mut [1, 2, 3, 0]);
    }
}
