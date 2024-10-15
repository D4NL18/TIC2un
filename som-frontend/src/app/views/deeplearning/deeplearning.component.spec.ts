import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DeeplearningComponent } from './deeplearning.component';

describe('DeeplearningComponent', () => {
  let component: DeeplearningComponent;
  let fixture: ComponentFixture<DeeplearningComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DeeplearningComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DeeplearningComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
