use spinoff::{spinners, Color, Spinner};

pub struct SilentableSpinner {
    spinner: Option<Spinner>,
}
impl SilentableSpinner {
    pub fn new(is_silent: bool, start_message: Option<impl Into<String>>) -> SilentableSpinner {
        if is_silent {
            Self { spinner: None }
        } else if start_message.is_some() {
            Self {
                spinner: Some(Spinner::new(
                    spinners::Dots,
                    start_message.unwrap().into(),
                    Color::Green,
                )),
            }
        // empty start message
        } else {
            Self {
                spinner: Some(Spinner::new(spinners::Dots, "", Color::Green)),
            }
        }
    }
    pub fn update(&mut self, message: impl Into<String>) {
        match &mut self.spinner {
            Some(spinner) => spinner.update(spinners::Dots, message.into(), Color::Blue),
            None => (),
        }
    }
    pub fn success(&mut self, message: &str) {
        match &mut self.spinner {
            Some(spinner) => spinner.success(message),
            None => (),
        }
    }
    pub fn fail(&mut self, message: &str) {
        match &mut self.spinner {
            Some(spinner) => spinner.fail(message),
            None => (),
        }
    }
}
